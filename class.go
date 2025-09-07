// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bufio"
	"compress/bzip2"
	"embed"
	"fmt"
	"io"
	"math/rand"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/pointlander/morpheus/kmeans"
)

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
		Word string
	}
	words := make([]*Vector[Line], 0, 8)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, " ")
		word := Vector[Line]{
			Meta: Line{
				Word: strings.TrimSpace(parts[0]),
			},
			Vector: make([]float32, 50),
		}
		for i, part := range parts[1:] {
			value, err := strconv.ParseFloat(strings.TrimSpace(part), 32)
			if err != nil {
				panic(err)
			}
			word.Vector[i] = float32(value)
		}
		words = append(words, &word)
	}
	index := make(map[string]*Vector[Line], len(words))
	for i := range words {
		index[words[i].Meta.Word] = words[i]
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

	parse := func(text string) []*Vector[Line] {
		reg := regexp.MustCompile(`\s+`)
		parts := reg.Split(text, -1)
		reg = regexp.MustCompile(`[\p{P}]+`)
		lines := make([]*Vector[Line], 0, 8)
		for _, part := range parts {
			part = reg.ReplaceAllString(part, "")
			word := index[strings.ToLower(part)]
			if word != nil {
				lines = append(lines, word)
			}
		}
		return lines
	}

	vectorize := func(linesA, linesB []*Vector[Line], seed int64) (Matrix[float64], Matrix[float64], []int) {
		lines := make([]*Vector[Line], len(linesA)+len(linesB))
		copy(lines[:len(linesA)], linesA)
		copy(lines[len(linesA):], linesB)

		config := Config{
			Iterations: 16,
			Size:       100,
			Divider:    1,
		}
		cov := Morpheus(seed, config, lines)

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
		lines := make([]*Vector[Line], len(fake0a)+len(fake1a))
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
				fmt.Println("h", clusters[ii], humana[ii].Meta.Word)
			}
			for ii := range v1.Rows {
				if clusters[ii+v0.Rows] == 0 {
					c++
				} else {
					d++
				}
				fmt.Println("m", clusters[ii+v0.Rows], lines[ii].Meta.Word)
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
