// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bufio"
	"compress/bzip2"
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

	"github.com/pointlander/morpheus/kmeans"
)

var (
	// FlagIris is the iris clusterting mode
	FlagIris = flag.Bool("iris", false, "iris clustering")
	// FlagClass classifies text
	FlagClass = flag.Bool("class", false, "classify text")
	// FlagText text generation mode
	FlagText = flag.Bool("text", false, "text generation mode")
	// FlagE is an experiment
	FlagE = flag.Bool("e", false, "experiment")
	// FlagTxt txt mode
	FlagTxt = flag.Bool("txt", false, "txt mode")
	// Flag3m markov mcts morpheus mode
	Flag3m = flag.Bool("3m", false, "markov mcts morpheus mode")
	// FlagExtreme extreme mode
	FlagExtreme = flag.Bool("extreme", false, "extreme mode")
	// FlagLearn learn the vector database
	FlagLearn = flag.Bool("learn", false, "learn the vector database")
	// FlagPrompt the prompt to use
	FlagPrompt = flag.String("prompt", "What is the meaning of life?", "the prompt to use")
	// cpuprofile profiles the program
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to `file`")
)

const (
	order = 4
)

type Markov [order]byte
type Model [order]map[Markov][]uint32

// Lookup looks a vector up
func Lookup(markov *[order]Markov, model *Model) []float32 {
	for i := range markov {
		i = order - 1 - i
		vector := model[i][markov[i]]
		if vector != nil {
			sum := float32(0.0)
			for _, value := range vector {
				sum += float32(value)
			}
			result := make([]float32, len(vector))
			for ii, value := range vector {
				result[ii] = float32(value) / sum
			}
			return result
		}
	}
	return nil
}

// Iterate iterates a markov model
func Iterate(markov *[order]Markov, state byte) {
	for i := range markov {
		state := state
		for ii, value := range markov[i][:i+1] {
			markov[i][ii], state = state, value
		}
	}
}

// 3mMode markov mcts morpheus mode
func _3mMode() {
	const (
		size     = 256
		width    = 256
		clusters = 4
		samples  = 1024
	)

	type File struct {
		Name string
		Data []byte
	}

	files := []File{
		{Name: "pg74.txt.bz2"},
		{Name: "10.txt.utf-8.bz2"},
		{Name: "76.txt.utf-8.bz2"},
		{Name: "84.txt.utf-8.bz2"},
		{Name: "100.txt.utf-8.bz2"},
		{Name: "1837.txt.utf-8.bz2"},
		{Name: "2701.txt.utf-8.bz2"},
		{Name: "3176.txt.utf-8.bz2"},
	}

	var books Model
	for i := range books {
		books[i] = make(map[Markov][]uint32)
	}
	var sets [clusters]Model
	for i := range sets {
		for ii := range sets[i] {
			sets[i][ii] = make(map[Markov][]uint32)
		}
	}
	load := func(book string) []byte {
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

		markov := [order]Markov{}
		for i, value := range data[:len(data)-6] {
			i += 3
			for ii := range markov {
				vector := books[ii][markov[ii]]
				if vector == nil {
					vector = make([]uint32, size)
				}
				vector[value]++
				books[ii][markov[ii]] = vector

				state := value
				for iii, value := range markov[ii][:ii+1] {
					markov[ii][iii], state = state, value
				}
			}
		}
		return data
	}
	for i := range files {
		files[i].Data = load(fmt.Sprintf("books/%s", files[i].Name))
	}
	for i := range books {
		fmt.Println(i, len(books[i]))
	}
	for i := 1; i < len(sets); i++ {
		sets[i] = books
	}

	type Segment struct {
		Segment []byte
		Rank    float64
		Cluster int
	}

	rng := rand.New(rand.NewSource(1))
	for range 3 {
		var list *Vector[Segment]
		count := 0
		input := []byte(*FlagPrompt)
		length := len(input) + width
		for _, vectors := range sets {
			for range 128 * samples {
				segment := Vector[Segment]{}
				markov := [order]Markov{}
				var val byte
				for _, val = range input {
					vector := Lookup(&markov, &vectors)
					if vector != nil {
						segment.Meta.Segment = append(segment.Meta.Segment, val)
						segment.Vector = append(segment.Vector, vector[val])
					}
					Iterate(&markov, val)
				}

				for range width {
					vector := Lookup(&markov, &vectors)
					book := Lookup(&markov, &books)
					if vector != nil {
						total, selection := float32(0.0), rng.Float32()
						for i, value := range vector {
							total += (value + book[i]) / 2
							if selection < total {
								segment.Meta.Segment = append(segment.Meta.Segment, byte(i))
								val = byte(i)
								segment.Vector = append(segment.Vector, (value+book[i])/2)
								break
							}
						}
					}
					Iterate(&markov, val)
				}
				for _, value := range segment.Vector {
					segment.Meta.Rank += float64(value)
				}
				if list == nil {
					list = &segment
					count++
					continue
				}
				iterator := list
				var prev *Vector[Segment]
				for iterator != nil {
					if iterator.Meta.Rank < segment.Meta.Rank {
						segment.Next = iterator
						count++
						if prev == nil {
							list = &segment
							break
						}
						prev.Next = &segment
						break
					}
					prev = iterator
					iterator = iterator.Next
				}
				if iterator == nil {
					prev.Next = &segment
					count++
				}
				if count > len(sets)*samples {
					iterator := list
					for iterator != nil && iterator.Next != nil && iterator.Next.Next != nil {
						iterator = iterator.Next
					}
					iterator.Next = nil
					count--
				}
			}
		}

		segments := []*Vector[Segment]{}
		iterator := list
		for iterator != nil {
			segments = append(segments, iterator)
			iterator = iterator.Next
		}
		sort.Slice(segments, func(i, j int) bool {
			return segments[i].Meta.Rank > segments[j].Meta.Rank
		})

		config := Config{
			Iterations: 8,
			Size:       length,
			Divider:    0,
		}

		cov := Morpheus(rng.Int63(), config, segments)
		for i := range cov {
			l2 := 0.0
			for _, value := range cov[i] {
				l2 += value * value
			}
			segments[i].Meta.Rank = math.Sqrt(l2)
		}

		meta := make([][]float64, len(segments))
		for i := range meta {
			meta[i] = make([]float64, len(segments))
		}
		const k = clusters
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
		for i := range segments {
			segments[i].Meta.Cluster = clusters[i]
		}

		sort.Slice(segments, func(i, j int) bool {
			return segments[i].Stddev < segments[j].Stddev
		})
		for i := range segments[:10] {
			fmt.Println(string(segments[i].Meta.Segment))
		}

		fmt.Println("---------------------------------------")

		for i := range sets {
			for ii := range sets[i] {
				sets[i][ii] = make(map[Markov][]uint32)
			}
		}
		for i := range segments {
			markov := [order]Markov{}
			for _, value := range segments[i].Meta.Segment {
				for iii := range markov {
					vector := sets[segments[i].Meta.Cluster][iii][markov[iii]]
					if vector == nil {
						vector = make([]uint32, size)
					}
					vector[value]++
					sets[segments[i].Meta.Cluster][iii][markov[iii]] = vector
					state := value
					for iv, value := range markov[iii][:iii+1] {
						markov[iii][iv], state = state, value
					}
				}
			}
		}
	}

	var set Model
	for i := range set {
		set[i] = make(map[Markov][]uint32)
	}
	for i := range sets {
		for ii := range sets[i] {
			for key, entry := range sets[i][ii] {
				vector := set[ii][key]
				if vector == nil {
					vector = make([]uint32, size)
				}
				for iii, value := range entry {
					vector[iii] += value
				}
				set[ii][key] = vector
			}
		}
	}

	segments := []*Vector[Segment]{}
	for range samples {
		input := []byte(*FlagPrompt)
		segment := Vector[Segment]{}
		markov := [order]Markov{}
		var val byte
		for _, val = range input {
			vector := Lookup(&markov, &set)
			if vector != nil {
				segment.Meta.Segment = append(segment.Meta.Segment, val)
				segment.Vector = append(segment.Vector, vector[val])
			}
			Iterate(&markov, val)
		}

		for range samples {
			vector := Lookup(&markov, &set)
			if vector != nil {
				total, selection := float32(0.0), rng.Float32()
				for i, value := range vector {
					total += value
					if selection < total {
						segment.Meta.Segment = append(segment.Meta.Segment, byte(i))
						val = byte(i)
						segment.Vector = append(segment.Vector, value)
						break
					}
				}
			}
			Iterate(&markov, val)
		}
		for _, value := range segment.Vector {
			segment.Meta.Rank += float64(value)
		}
		segments = append(segments, &segment)
	}
	sort.Slice(segments, func(i, j int) bool {
		return segments[i].Meta.Rank > segments[j].Meta.Rank
	})
	fmt.Println(string(segments[0].Meta.Segment))
}

// ExtremeMode extreme mode
func ExtremeMode() {
	const (
		size = 256
	)

	type File struct {
		Name  string
		Data  []byte
		Model Model
	}

	files := []File{
		{Name: "pg74.txt.bz2"},
		{Name: "10.txt.utf-8.bz2"},
		{Name: "76.txt.utf-8.bz2"},
		{Name: "84.txt.utf-8.bz2"},
		{Name: "100.txt.utf-8.bz2"},
		{Name: "1837.txt.utf-8.bz2"},
		{Name: "2701.txt.utf-8.bz2"},
		{Name: "3176.txt.utf-8.bz2"},
	}

	load := func(book *File) {
		path := fmt.Sprintf("books/%s", book.Name)
		file, err := Text.Open(path)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		data, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}

		markov := [order]Markov{}
		for i := range book.Model {
			book.Model[i] = make(map[Markov][]uint32)
		}
		for _, value := range data {
			for ii := range markov {
				vector := book.Model[ii][markov[ii]]
				if vector == nil {
					vector = make([]uint32, size)
				}
				vector[value]++
				book.Model[ii][markov[ii]] = vector

				state := value
				for iii, value := range markov[ii][:ii+1] {
					markov[ii][iii], state = state, value
				}
			}
		}
		book.Data = data
	}
	for i := range files {
		load(&files[i])
		fmt.Println(files[i].Name)
		for ii := range files[i].Model {
			fmt.Println(len(files[i].Model[ii]))
		}
	}

	rng := rand.New(rand.NewSource(1))
	max, result := float32(0.0), ""
	for range 64 * 1024 {
		markov, data := [order]Markov{}, []byte{}
		for _, symbol := range []byte("What is the meaning of life?") {
			Iterate(&markov, symbol)
		}
		sum := float32(0.0)
		for range 1024 {
			book := rng.Intn(len(files))
			vector := Lookup(&markov, &files[book].Model)
			if vector != nil {
				total, selection := float32(0.0), rng.Float32()
				for i, value := range vector {
					total += value
					if selection < total {
						data = append(data, byte(i))
						sum += value
						Iterate(&markov, byte(i))
						break
					}
				}
			}
		}
		if sum > max {
			max, result = sum, string(data)
		}
	}
	fmt.Println(result)
}

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

	if *FlagText {
		TextMode()
		return
	}

	if *FlagTxt {
		TxtMode()
		return
	}

	if *Flag3m {
		_3mMode()
		return
	}

	if *FlagExtreme {
		ExtremeMode()
		return
	}

	type File struct {
		Name string
		Data []byte
	}

	files := []File{
		{Name: "pg74.txt.bz2"},
		{Name: "10.txt.utf-8.bz2"},
		{Name: "76.txt.utf-8.bz2"},
		{Name: "84.txt.utf-8.bz2"},
		{Name: "100.txt.utf-8.bz2"},
		{Name: "1837.txt.utf-8.bz2"},
		{Name: "2701.txt.utf-8.bz2"},
		{Name: "3176.txt.utf-8.bz2"},
	}
	load := func(book string) []byte {
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
		return data
	}
	for i := range files {
		files[i].Data = load(fmt.Sprintf("books/%s", files[i].Name))
	}

	bible := string(files[1].Data)
	reg := regexp.MustCompile(`\s+`)
	parts := reg.Split(bible, -1)
	reg = regexp.MustCompile(`[\p{P}]+`)
	unique := make(map[string]int)
	for _, part := range parts {
		part = reg.ReplaceAllString(part, "")
		_, err := strconv.Atoi(part)
		if err == nil {
			continue
		}
		count := unique[strings.ToLower(part)]
		count++
		unique[strings.ToLower(part)] = count
	}
	type Pair struct {
		Word  string
		Count int
	}
	pairs := make([]Pair, len(unique))
	c := 0
	for word, count := range unique {
		pairs[c].Word = word
		pairs[c].Count = count
		c++
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Count > pairs[j].Count
	})
	fmt.Println(len(pairs))
	for i := range pairs[:1024] {
		fmt.Println(pairs[i].Count, pairs[i].Word)
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
		Word    string
		Cluster int
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

	//selection := []string{"true", "false", "god", "jesus", "faith",
	//"truth", "atheism", "philosophy", "lord", "savior", "good", "evil"}
	pairs = pairs[:1024]
	vectors := make([]*Vector[Line], 0, len(pairs))
	for i := range pairs {
		vector := index[pairs[i].Word]
		if vector != nil {
			vectors = append(vectors, vector)
		}
	}

	fmt.Println("cosine similarity")
	for i := range vectors {
		a := NewMatrix(50, 1, vectors[i].Vector...)
		for ii := range vectors {
			b := NewMatrix(50, 1, vectors[ii].Vector...)
			cs := a.CS(b)
			_ = cs
			//fmt.Printf("%.8f ", cs)
		}
		//fmt.Println()
	}

	config := Config{
		Iterations: 8,
		Size:       50,
		Divider:    0,
	}
	fmt.Println("standard deviation")
	cov := Morpheus(1, config, vectors)
	for i := range vectors {
		fmt.Println(vectors[i].Stddev, vectors[i].Meta.Word)
	}
	fmt.Println("cov")
	/*for i := range vectors {
		for ii := range vectors {
			fmt.Printf("%.8f ", cov[i][ii])
		}
		fmt.Println()
	}*/
	fmt.Println("correlation")
	/*for i := range vectors {
		for ii := range vectors {
			fmt.Printf("%.8f ", cov[i][ii]/(vectors[i].Stddev*vectors[ii].Stddev))
		}
		fmt.Println()
	}*/
	meta := make([][]float64, len(vectors))
	for i := range meta {
		meta[i] = make([]float64, len(vectors))
	}
	const k = 2
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
	for i := range vectors {
		vectors[i].Meta.Cluster = clusters[i]
	}
	fmt.Println("clustering")
	for i := range vectors {
		fmt.Println(vectors[i].Meta.Cluster, vectors[i].Meta.Word)
	}
}
