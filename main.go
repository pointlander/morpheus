// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"compress/bzip2"
	"embed"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/alixaxel/pagerank"
	"github.com/pointlander/morpheus/kmeans"
)

//go:embed iris.zip
var Iris embed.FS

//go:embed pg74.txt.bz2
var Text embed.FS

const FakeText0 = `The Peculiar Case of Mr. Hiram Sneed
It was a fine, dry morning when I met Mr. Hiram Sneed—at least, that’s what I call him, though I don’t reckon he’d answer to that if you asked him direct. But as I’ll explain, it wasn’t so much the name that set him apart as it was his general disposition.
I had been sitting on the porch of the old hotel, rocking in the sun and thinking that if the weather held, I might just take to my hammock for the day, when Hiram came wandering up the street. He was a big man, broad as a barn door, and he had a face like a mix of a bulldog and a catfish—tough but oddly saggy, like something had been trying to escape from it and hadn’t quite succeeded.
Now, most folks might’ve taken one look and crossed the street, but not me. No, sir. I’d learned a long time ago that the best way to get a straight answer out of a man was to ask him a foolish question. So, before he could make his way into the hotel, I hailed him.
“Say, Hiram,” I shouted, “do you reckon a man can be too rich for his own good?”
He paused, eyeing me suspiciously, but then a smile cracked across his face like a dry log breaking open. He came on over.
“Well, now,” he said, “that’s an interesting question. I reckon a man can have all sorts of riches, but if he ain't got no sense to enjoy 'em, I don't see much use in it. Take old Miss Dobbins down yonder. She’s got more money than a river has fish, but she’ll spend a whole afternoon fretting over whether her curtains are too lavender for the parlor.”
I nodded thoughtfully, but I could tell he wasn’t finished.
“Riches,” he continued, “ain’t worth a lick if you ain't got the sense to know what to do with ‘em. I seen a feller once—young Buck Trumble, name was—who got himself a hundred acres of land from his uncle. A hundred acres, mind you! But when he found out it was mostly swamp, he just sat right down in the mud and cried like a newborn calf.”
I laughed, mostly to myself, but I had to ask him more.
“Was he really that upset?”
“Well, not at first,” Hiram said, “but when he figured out there wasn’t no dry land to build his house on, he started making himself a bit of a spectacle. I heard him hollering all the way from the creek, cursing the trees and the snakes, saying that the swamp had cursed him, and he was cursed in return. But I reckon the swamp wasn’t at fault—he was just too stubborn to see the whole picture.”
I raised an eyebrow. “And what’d you tell him?”
“Oh, I told him to take his boots off, roll up his pants, and take a walk through it. He did, and by the time he got to the far side, he had himself a mind full of new ideas. That swamp, you see, wasn’t a curse. It was an opportunity. He’s a successful businessman now, selling swamp muck for fertilizer.”
I couldn’t help but laugh again. “So, all that fuss for some muck?”
Hiram chuckled too. “Well, sure. The way I see it, a feller can spend his whole life pining after a dry patch of land, or he can make himself a fortune out of something no one else would touch. It’s all about how you look at things.”
I leaned back in my chair, scratching my chin. “I reckon you might be right. Maybe I’ve been too busy looking for my own dry land.”
Hiram’s smile faded a little, but he tipped his hat. “I reckon you have. But, sometimes, it’s best to look at the swamp with a little less disgust. You might be surprised by what you find.”
And with that, he turned and walked off down the road, leaving me to wonder if I should go searching for a swamp or if maybe I’d just stick to my porch a while longer.`

const FakeText1 = `A Riverbank Tale
It was a Tuesday hotter than a stove lid, and the Mississippi lay stretched out like a big, lazy dog that didn’t aim to move till winter. Down at the bend, young Lem Haskins was trying to coax his mule into the water, for reasons known only to himself and, perhaps, the mule—though the mule wasn’t talkative on the matter.
“Git in there, Clyde,” Lem said, tugging on the rope. “If you’re fixin’ to die of thirst one day, don’t let it be within sneezin’ distance of the river.”
Clyde flicked his tail and looked around with a solemn air, as though he had business far too dignified to be concerned with a boy’s schemes.
About then, Old Man Perkins came limping down the path, pipe in mouth, cane thumping. He stopped to watch the contest between boy and beast.
“Lem,” he drawled, “that mule knows more about water than you do about sense. If he ain’t going in, it’s ‘cause he reckons the river’s got plans you wouldn’t care to meet.”
Lem puffed up, red-faced. “Mister Perkins, I reckon Clyde here don’t know beans about plans, ‘cept where to find ‘em in a feed sack.”
The mule brayed so loud it echoed up the trees, like he was laughing at both of them. And I’ll be switched if, the very next minute, a big old log floated past, with a snake coiled up on top, mean as the devil’s own whip.
Lem went quiet. Clyde stamped a hoof, turned his ears, and gave Lem a look that said as plain as words: Next time, boy, maybe listen to the mule.
Old Man Perkins puffed his pipe, grinning around it. “Well now,” he said, “that’s the trouble with youth—always tryin’ to teach wisdom somethin’ new.”
And the river rolled on, chuckling at the whole spectacle, like it had seen the same story play out a thousand times before and planned to see it a thousand times again.`

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Cluster  int
	Index    int
	Rank     float64
	AE       int
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Inverse is the labels inverse map
var Inverse = [3]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

var (
	// FlagIris is the iris clusterting mode
	FlagIris = flag.Bool("iris", false, "iris clustering")
)

// IrisMode is the iris clustering mode
func IrisMode() {
	iris := Load()
	rng := rand.New(rand.NewSource(1))
	const iterations = 512
	results := make([][]float64, iterations)
	for iteration := range iterations {
		a, b := NewMatrix(4, 4, make([]float64, 4*4)...), NewMatrix(4, 4, make([]float64, 4*4)...)
		index := 0
		for range a.Rows {
			for range a.Cols {
				a.Data[index] = rng.NormFloat64()
				b.Data[index] = rng.NormFloat64()
				index++
			}
		}
		a = a.Softmax(1)
		b = b.Softmax(1)
		graph := pagerank.NewGraph()
		for i := range iris {
			for ii := range iris {
				x, y := NewMatrix(4, 1, make([]float64, 4)...), NewMatrix(4, 1, make([]float64, 4)...)
				for i, value := range iris[i].Measures {
					x.Data[i] = value
				}
				for i, value := range iris[ii].Measures {
					y.Data[i] = value
				}
				x = a.MulT(x)
				y = b.MulT(y)
				cs := x.CS(y)
				graph.Link(uint32(i), uint32(ii), cs)
			}
		}
		result := make([]float64, len(iris))
		graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
			result[node] = rank
		})
		results[iteration] = result
	}
	avg := make([]float64, len(iris))
	for _, result := range results {
		for i, value := range result {
			avg[i] += value
		}
	}
	for i, value := range avg {
		avg[i] = value / float64(iterations)
	}
	stddev := make([]float64, len(iris))
	for _, result := range results {
		for i, value := range result {
			diff := value - avg[i]
			stddev[i] += diff * diff
		}
	}
	for i, value := range stddev {
		stddev[i] = math.Sqrt(value / float64(iterations))
	}
	for i := range iris {
		iris[i].Rank = stddev[i]
	}
	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Rank < iris[j].Rank
	})
	for i := range stddev {
		fmt.Println(iris[i].Label)
	}

	cov := make([][]float64, len(iris))
	for i := range cov {
		cov[i] = make([]float64, len(iris))
	}
	for _, measures := range results {
		for i, v := range measures {
			for ii, vv := range measures {
				diff1 := avg[i] - v
				diff2 := avg[ii] - vv
				cov[i][ii] += diff1 * diff2
			}
		}
	}
	if len(results) > 0 {
		for i := range cov {
			for ii := range cov[i] {
				cov[i][ii] = cov[i][ii] / float64(len(results))
			}
		}
	}
	fmt.Println("K=")
	for i := range cov {
		fmt.Println(cov[i])
	}
	fmt.Println("u=")
	fmt.Println(avg)
	fmt.Println()

	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Index < iris[j].Index
	})

	meta := make([][]float64, len(iris))
	for i := range meta {
		meta[i] = make([]float64, len(iris))
	}
	const k = 3
	for i := 0; i < 33; i++ {
		clusters, _, err := kmeans.Kmeans(int64(i+1), cov, k, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}
		for i := 0; i < len(meta); i++ {
			target := clusters[i]
			for j, v := range clusters {
				if v == target {
					meta[i][j]++
				}
			}
		}
	}
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, value := range clusters {
		iris[i].Cluster = value
	}
	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Cluster < iris[j].Cluster
	})
	for i := range stddev {
		fmt.Println(iris[i].Cluster, iris[i].Label)
	}
	a := make(map[string][3]int)
	for i := range iris {
		histogram := a[iris[i].Label]
		histogram[iris[i].Cluster]++
		a[iris[i].Label] = histogram
	}
	for k, v := range a {
		fmt.Println(k, v)
	}

	var auto [3]*AutoEncoder
	for i := range auto {
		auto[i] = NewAutoEncoder(len(iris), 1)
	}
	for i := range cov {
		sum := 0.0
		for _, value := range cov[i] {
			sum += value
		}
		for ii, value := range cov[i] {
			cov[i][ii] = value / sum
		}
	}
	for range 32 {
		var histogram [3]int
		for i := range cov {
			min, minIndex := math.MaxFloat32, 0
			for ii := range auto {
				e := auto[ii].Measure(cov[i], cov[i])
				if e < min {
					min, minIndex = e, ii
				}
			}
			histogram[minIndex]++
		}
		fmt.Println(histogram)
		min, minIndex := math.MaxInt64, 0
		for i, value := range histogram {
			if value < min {
				min, minIndex = value, i
			}
		}
		perm := rng.Perm(len(cov))
		for i := range cov {
			i = perm[i]
			auto[minIndex].Encode(cov[i], cov[i])
		}
	}
}

func main() {
	flag.Parse()

	if *FlagIris {
		IrisMode()
		return
	}

	reader, err := zip.OpenReader("glove.2024.wikigiga.50d.zip")
	if err != nil {
		panic(err)
	}
	defer reader.Close()
	input, err := reader.File[0].Open()
	if err != nil {
		panic(err)
	}
	defer input.Close()
	scanner := bufio.NewScanner(input)
	type Line struct {
		Word   string
		Vector [50]float32
	}
	words := make([]Line, 0, 8)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, " ")
		word := Line{
			Word: strings.TrimSpace(parts[0]),
		}
		for i, part := range parts[1:] {
			value, err := strconv.ParseFloat(strings.TrimSpace(part), 32)
			if err != nil {
				panic(err)
			}
			word.Vector[i] = float32(value)
		}
		words = append(words, word)
	}
	index := make(map[string]*Line, len(words))
	for i := range words {
		index[words[i].Word] = &words[i]
	}

	file, err := Text.Open("pg74.txt.bz2")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	breader := bzip2.NewReader(file)
	data, err := io.ReadAll(breader)
	if err != nil {
		panic(err)
	}

	parse := func(text string) []*Line {
		reg := regexp.MustCompile(`\s+`)
		parts := reg.Split(text, -1)
		reg = regexp.MustCompile(`[\p{P}]+`)
		lines := make([]*Line, 0, 8)
		for _, part := range parts {
			part = reg.ReplaceAllString(part, "")
			word := index[strings.ToLower(part)]
			if word != nil {
				lines = append(lines, word)
			}
		}
		return lines
	}

	vectorize := func(lines []*Line) Matrix[float64] {
		rng := rand.New(rand.NewSource(1))
		const iterations = 8
		results := make([][]float64, iterations)
		for iteration := range iterations {
			a, b := NewMatrix(50, 50, make([]float64, 50*50)...), NewMatrix(50, 50, make([]float64, 50*50)...)
			index := 0
			for range a.Rows {
				for range a.Cols {
					a.Data[index] = rng.NormFloat64()
					b.Data[index] = rng.NormFloat64()
					index++
				}
			}
			a = a.Softmax(1)
			b = b.Softmax(1)
			graph := pagerank.NewGraph()
			for i := range lines {
				for ii := range lines {
					x, y := NewMatrix(50, 1, make([]float64, 50)...), NewMatrix(50, 1, make([]float64, 50)...)
					for i, value := range lines[i].Vector {
						if value < 0 {
							value = -value
						}
						x.Data[i] = float64(value)
					}
					for i, value := range lines[ii].Vector {
						if value < 0 {
							value = -value
						}
						y.Data[i] = float64(value)
					}
					x = a.MulT(x)
					y = b.MulT(y)
					cs := x.CS(y)
					graph.Link(uint32(i), uint32(ii), cs)
				}
			}
			result := make([]float64, len(lines))
			graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
				result[node] = rank
			})
			results[iteration] = result
		}
		avg := make([]float64, len(lines))
		for _, result := range results {
			for i, value := range result {
				avg[i] += value
			}
		}
		for i, value := range avg {
			avg[i] = value / float64(iterations)
		}

		cov := make([][]float64, len(lines))
		for i := range cov {
			cov[i] = make([]float64, len(lines))
		}
		for _, measures := range results {
			for i, v := range measures {
				for ii, vv := range measures {
					diff1 := avg[i] - v
					diff2 := avg[ii] - vv
					cov[i][ii] += diff1 * diff2
				}
			}
		}
		if len(results) > 0 {
			for i := range cov {
				for ii := range cov[i] {
					cov[i][ii] = cov[i][ii] / float64(len(results))
				}
			}
		}
		embedding := NewMatrix(len(lines), len(lines), make([]float64, len(lines)*len(lines))...)
		for i := range cov {
			for ii, value := range cov[i] {
				embedding.Data[i*len(lines)+ii] = value
			}
		}
		return embedding
	}

	cs0, cs1, cs2 := 0.0, 0.0, 0.0
	rng := rand.New(rand.NewSource(1))
	human := parse(string(data))
	fake0 := parse(FakeText0)
	fake1 := parse(FakeText1)
	const samples = 64
	for i := range samples {
		size := rng.Intn(50) + 50
		fmt.Println(i)
		index := rng.Intn(len(human) - size)
		human := human[index : index+size]
		index = rng.Intn(len(fake0) - size)
		fake0 := fake0[index : index+size]
		index = rng.Intn(len(fake1) - size)
		fake1 := fake1[index : index+size]
		vhuman := vectorize(human)
		vfake0 := vectorize(fake0)
		vfake1 := vectorize(fake1)
		cs0 += vhuman.CS(vfake0)
		cs1 += vhuman.CS(vfake1)
		cs2 += vfake0.CS(vfake1)
	}
	fmt.Println("human vs fake0", cs0/float64(samples))
	fmt.Println("human vs fake1", cs1/float64(samples))
	fmt.Println("fake0 vs fake1", cs2/float64(samples))
}
