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
	"os"
	"regexp"
	"runtime/pprof"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/alixaxel/pagerank"
	"github.com/pointlander/morpheus/kmeans"
)

//go:embed iris.zip
var Iris embed.FS

//go:embed books/*
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
	// FlagClass classifies text
	FlagClass = flag.Bool("class", false, "classify text")
	// FlagE is an experiment
	FlagE = flag.Bool("e", false, "experiment")
	// cpuprofile profiles the program
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to `file`")
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

// ClassMode is the text classification mode
func ClassMode() {
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

	file, err := Text.Open("books/pg74.txt.bz2")
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

	vectorize := func(linesA, linesB []*Line, seed int64) (Matrix[float64], Matrix[float64], []int) {
		lines := make([]*Line, len(linesA)+len(linesB))
		copy(lines[:len(linesA)], linesA)
		copy(lines[len(linesA):], linesB)
		rng := rand.New(rand.NewSource(seed))
		const iterations = 16
		results := make([][]float64, iterations)
		for iteration := range iterations {
			a, b := NewMatrix(100, 100, make([]float64, 100*100)...), NewMatrix(100, 100, make([]float64, 100*100)...)
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
					x, y := NewMatrix(100, 1, make([]float64, 100)...), NewMatrix(100, 1, make([]float64, 100)...)
					for i, value := range lines[i].Vector {
						if value < 0 {
							x.Data[50+i] = float64(-value)
							continue
						}
						x.Data[i] = float64(value)
					}
					for i, value := range lines[ii].Vector {
						if value < 0 {
							y.Data[50+i] = float64(-value)
							continue
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

		meta := make([][]float64, len(lines))
		for i := range meta {
			meta[i] = make([]float64, len(lines))
		}
		k := 2
		if *FlagE {
			k = 2
		}
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
		clusters, _, err := kmeans.Kmeans(1, meta, k, kmeans.SquaredEuclideanDistance, -1)
		if err != nil {
			panic(err)
		}

		embedding := NewMatrix(len(lines), len(linesA), make([]float64, len(lines)*len(linesA))...)
		for i := range cov[:len(linesA)] {
			for ii, value := range cov[i] {
				embedding.Data[i*len(lines)+ii] = value
			}
		}
		embedding1 := NewMatrix(len(lines), len(linesB), make([]float64, len(lines)*len(linesB))...)
		for i := range cov[len(linesA):] {
			for ii, value := range cov[i+len(linesA)] {
				embedding1.Data[i*len(lines)+ii] = value
			}
		}
		return embedding, embedding1, clusters
	}

	var cs [7]float64
	var diff [7][2]float64
	rng := rand.New(rand.NewSource(1))
	human := parse(string(data))
	fake0 := parse(FakeText0)
	fake1 := parse(FakeText1)
	const samples = 16
	for i := range samples {
		size := rng.Intn(50) + 50
		if *FlagE {
			size = 150
		}
		fmt.Println(i)
		index := rng.Intn(len(human) - size)
		humana := human[index : index+size]
		index = rng.Intn(len(fake0) - size)
		fake0a := fake0[index : index+size]
		index = rng.Intn(len(fake1) - size)
		fake1a := fake1[index : index+size]
		index = rng.Intn(len(human) - size)
		humanb := human[index : index+size]
		index = rng.Intn(len(fake0) - size)
		fake0b := fake0[index : index+size]
		index = rng.Intn(len(fake1) - size)
		fake1b := fake1[index : index+size]
		lines := make([]*Line, len(fake0a)+len(fake1a))
		copy(lines[:len(fake0a)], fake0a)
		copy(lines[len(fake0a):], fake1a)
		var wg sync.WaitGroup
		var (
			v        [14]Matrix[float64]
			clusters [7][]int
		)
		seeds := make([]int64, 7)
		for ii := range seeds {
			seeds[ii] = rng.Int63()
		}
		if *FlagE {
			v0, v1, clusters := vectorize(humana, lines, seeds[6])

			a, b, c, d := 0, 0, 0, 0
			for ii := range v0.Rows {
				if clusters[ii] == 0 {
					a++
				} else {
					b++
				}
				fmt.Println("h", clusters[ii], humana[ii].Word)
			}
			for ii := range v1.Rows {
				if clusters[ii+v0.Rows] == 0 {
					c++
				} else {
					d++
				}
				fmt.Println("m", clusters[ii+v0.Rows], lines[ii].Word)
			}
			df := a - c
			if df < 0 {
				df = -df
			}
			diff[0][0] += float64(df)
			df = b - d
			if df < 0 {
				df = -df
			}
			diff[0][1] += float64(df)
			fmt.Println(a, b, c, d)
			fmt.Println(diff[0])
			e1 := NewMatrix(v1.Cols, len(fake0a), make([]float64, v1.Cols*len(fake0a))...)
			copy(e1.Data, v1.Data[:v1.Cols*len(fake0a)])
			e2 := NewMatrix(v1.Cols, len(fake1a), make([]float64, v1.Cols*len(fake1a))...)
			copy(e2.Data, v1.Data[v1.Cols*len(fake0a):])
			fmt.Println(v0.CS(e1))
			fmt.Println(v0.CS(e2))
			fmt.Println(e1.CS(e2))
			v0.Cols, v0.Rows = v0.Cols*v0.Rows, 1
			e1.Cols, e1.Rows = e1.Cols*e1.Rows, 1
			e2.Cols, e2.Rows = e2.Cols*e2.Rows, 1
			fmt.Println(v0.CS(e1))
			fmt.Println(v0.CS(e2))
			fmt.Println(e1.CS(e2))
			return
		}
		wg.Go(func() { v[0], v[1], clusters[0] = vectorize(humana, fake0a, seeds[0]) })
		wg.Go(func() { v[2], v[3], clusters[1] = vectorize(humana, fake1a, seeds[1]) })
		wg.Go(func() { v[4], v[5], clusters[2] = vectorize(fake0a, fake1a, seeds[2]) })
		wg.Go(func() { v[6], v[7], clusters[3] = vectorize(humana, humanb, seeds[3]) })
		wg.Go(func() { v[8], v[9], clusters[4] = vectorize(fake0a, fake0b, seeds[4]) })
		wg.Go(func() { v[10], v[11], clusters[5] = vectorize(fake1a, fake1b, seeds[5]) })
		wg.Go(func() { v[12], v[13], clusters[6] = vectorize(humana, lines, seeds[6]) })
		wg.Wait()

		c := 0
		for i := 0; i < len(v); i += 2 {
			cs[c] += v[i].CS(v[i+1])
			c++
		}

		x := 0
		for i := 0; i < len(v); i += 2 {
			a, b, c, d := 0, 0, 0, 0
			for ii := range v[i].Rows {
				if clusters[x][ii] == 0 {
					a++
				} else {
					b++
				}
			}
			for ii := range v[i+1].Rows {
				if clusters[x][ii+v[i].Rows] == 0 {
					c++
				} else {
					d++
				}
			}
			df := a - c
			if df < 0 {
				df = -df
			}
			diff[x][0] += float64(df)
			df = b - d
			if df < 0 {
				df = -df
			}
			diff[x][1] += float64(df)
			x++
		}
	}
	fmt.Println("human vs fake0", cs[0]/float64(samples))
	fmt.Println("human vs fake1", cs[1]/float64(samples))
	fmt.Println("fake0 vs fake1", cs[2]/float64(samples))
	fmt.Println("human", cs[3]/float64(samples))
	fmt.Println("fake0", cs[4]/float64(samples))
	fmt.Println("fake1", cs[5]/float64(samples))
	fmt.Println("human vs fake0 & fake1", cs[6]/float64(samples))
	fmt.Println(diff)
}

// Markov is a markov state
type Markov [2]byte

func main() {
	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			panic(err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			panic(err)
		}
		defer pprof.StopCPUProfile()
	}

	if *FlagIris {
		IrisMode()
		return
	}

	if *FlagClass {
		ClassMode()
		return
	}

	vectors := make(map[Markov][]uint32)
	load := func(book string) {
		file, err := Text.Open(book)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		data, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}

		markov := Markov{}
		for i, value := range data[:len(data)-6] {
			i += 3
			vector := vectors[markov]
			if vector == nil {
				vector = make([]uint32, 256)
			}
			vector[data[i-3]]++
			vector[data[i-1]]++
			vector[value]++
			vector[data[i+1]]++
			vector[data[i+3]]++
			vectors[markov] = vector
			state := value
			for i, value := range markov {
				markov[i], state = state, value
			}
		}
	}
	load("books/pg74.txt.bz2")
	load("books/76.txt.utf-8.bz2")
	load("books/1837.txt.utf-8.bz2")
	load("books/3176.txt.utf-8.bz2")
	fmt.Println(len(vectors))

	vectorize := func(input string, seed int64) string {
		type Line struct {
			Symbol byte
			Vector []float32
		}
		markov := Markov{}
		lines := make([]*Line, 0, 8)
		for _, value := range []byte(input) {
			line := Line{
				Symbol: value,
				Vector: make([]float32, 256),
			}
			vector := vectors[markov]
			if vector == nil {
				markov := markov
				for ii := range 256 {
					markov[0] = byte(ii)
					for ii, value := range vectors[markov] {
						line.Vector[ii] += float32(value)
					}
				}
			} else {
				for ii, value := range vector {
					line.Vector[ii] = float32(value)
				}
			}
			state := value
			for ii, value := range markov {
				markov[ii], state = state, value
			}
			lines = append(lines, &line)
		}
		count := 0
		for ii := range 256 {
			markov := markov
			state := byte(ii)
			for ii, value := range markov {
				markov[ii], state = state, value
			}
			vector := vectors[markov]
			if vector != nil {
				count++
				line := Line{
					Symbol: byte(ii),
					Vector: make([]float32, 256),
				}
				for ii, value := range vector {
					line.Vector[ii] = float32(value)
				}
				lines = append(lines, &line)
			}
		}
		fmt.Println(len(lines), count)

		for i := range lines {
			sum := float32(0.0)
			for _, value := range lines[i].Vector {
				sum += value
			}
			for ii, value := range lines[i].Vector {
				lines[i].Vector[ii] = value / sum
			}
		}

		rng := rand.New(rand.NewSource(seed))
		const (
			iterations = 32
			size       = 256
		)
		results := make([][]float64, iterations)
		for iteration := range iterations {
			a, b := NewMatrix(size, size, make([]float32, size*size)...), NewMatrix(size, size, make([]float32, size*size)...)
			index := 0
			for range a.Rows {
				for range a.Cols {
					a.Data[index] = float32(rng.NormFloat64())
					b.Data[index] = float32(rng.NormFloat64())
					index++
				}
			}
			a = a.Softmax(1)
			b = b.Softmax(1)
			graph := pagerank.NewGraph()
			for i := range lines {
				x := NewMatrix(size, 1, make([]float32, size)...)
				for ii, value := range lines[i].Vector {
					if value < 0 {
						x.Data[ii] = -value
						continue
					}
					x.Data[ii] = value
				}
				xx := a.MulT(x)
				for ii := range lines {
					y := NewMatrix(size, 1, make([]float32, size)...)
					for iii, value := range lines[ii].Vector {
						if value < 0 {
							y.Data[iii] = -value
							continue
						}
						y.Data[iii] = value
					}
					yy := b.MulT(y)
					cs := xx.CS(yy)
					graph.Link(uint32(i), uint32(ii), float64(cs))
				}
			}
			result := make([]float64, len(lines))
			graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
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

		stddev := make([]float64, len(lines))
		for _, result := range results {
			for i, value := range result {
				diff := value - avg[i]
				stddev[i] += diff * diff
			}
		}
		for i, value := range stddev {
			stddev[i] = math.Sqrt(value / float64(iterations))
		}

		/*cov := make([][]float64, len(lines))
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
		}*/

		sum, norm, c := 0.0, make([]float64, count), 0
		for i := len(lines) - count; i < len(lines); i++ {
			sum += stddev[i]
		}
		for i := len(lines) - count; i < len(lines); i++ {
			norm[c] = sum / stddev[i]
			c++
		}
		softmax(norm)

		total, selected, index := 0.0, rng.Float64(), 0
		for i := range norm {
			total += norm[i]
			if selected < total {
				index = i
				break
			}
		}

		next := []byte(input)
		next = append(next, lines[(len(lines)-count)+index].Symbol)

		return string(next)
	}

	rng := rand.New(rand.NewSource(1))
	state := "The old lady pulled her"
	for range 33 {
		state = vectorize(state, rng.Int63())
		fmt.Println(state)
	}
}
