// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"flag"
	"fmt"
	"github.com/alixaxel/pagerank"
	"io"
	"math/rand"
	"os"
	"regexp"
	"runtime/pprof"
	"sort"
	"strings"
)

var (
	// FlagIris is the iris clusterting mode
	FlagIris = flag.Bool("iris", false, "iris clustering")
	// FlagIrisMarkov is the iris markov model
	FlagIrisMarkov = flag.Bool("irismarkov", false, "iris markov mode")
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
	// FlagMach1 mach1 mode
	FlagMach1 = flag.Bool("mach1", false, "mach 1 mode")
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

	if *FlagIrisMarkov {
		IrisMarkovMode()
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

	if *FlagMach1 {
		Mach1Mode()
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

	type Word struct {
		Word  string
		Count int
	}

	bible := string(files[1].Data)
	reg := regexp.MustCompile(`\s+`)
	parts := reg.Split(bible, -1)
	reg = regexp.MustCompile(`[\p{P}]+|[\d]`)
	unique := make(map[string]*Word)
	links := make(map[string]map[string]uint64)
	previous, current := "", ""
	for _, part := range parts {
		parts := reg.Split(part, -1)
		parts = append(parts, reg.FindAllString(part, -1)...)
		for _, part := range parts {
			word := strings.ToLower(strings.TrimSpace(part))
			count := unique[word]
			if count == nil {
				count = &Word{
					Word: word,
				}
			}
			count.Count++
			unique[word] = count

			previous, current = current, word
			link := links[previous]
			if link == nil {
				link = make(map[string]uint64)
			}
			link[current]++
			links[previous] = link
		}
	}
	words := make([]*Word, len(unique))
	{
		count := 0
		for _, word := range unique {
			words[count] = word
			count++
		}
	}
	sort.Slice(words, func(i, j int) bool {
		if words[i].Count == words[j].Count {
			return words[i].Word < words[j].Word
		}
		return words[i].Count > words[j].Count
	})
	fmt.Println(len(words))
	length := 1 * 1024
	words = words[:length]

	type Path struct {
		Path []int
		Cost float64
	}
	done := make(chan Path)
	process := func(seed int64) {
		context, cost := []int{}, 0.0
		for _, word := range []string{"the", "lord", "is", "good"} {
			for i := range words {
				if words[i].Word == word {
					context = append(context, i)
					break
				}
			}
		}

		rng := rand.New(rand.NewSource(seed))
		for range 33 {
			size := len(words) + len(context)
			graph := pagerank.NewGraph()
			adjacency := NewMatrix(size, size, make([]float32, size*size)...)
			for i := range words {
				from := links[words[i].Word]
				for ii := range words {
					to := from[words[ii].Word]
					graph.Link(uint32(i), uint32(ii), float64(to))
					adjacency.Data[i*adjacency.Cols+ii] = float32(to)
				}
			}
			for i, word := range context {
				for ii := range words {
					sum := float32(0.0)
					for _, value := range adjacency.Data[ii*adjacency.Cols : ii*adjacency.Cols+length] {
						sum += value
					}
					graph.Link(uint32(ii), uint32(length+i), float64(sum/8))
					adjacency.Data[ii*adjacency.Cols+length+i] += sum / 8
				}
				copy(adjacency.Data[(length+i)*adjacency.Cols:(length+i+1)*adjacency.Cols],
					adjacency.Data[word*adjacency.Cols:(word+1)*adjacency.Cols])
				if i > 0 {
					sum := float32(0.0)
					for _, value := range adjacency.Data[(length+i-1)*adjacency.Cols : (length+i-1)*adjacency.Cols+length] {
						sum += value
					}
					graph.Link(uint32(length+i-1), uint32(length+i), float64(sum/8))
					adjacency.Data[(length+i-1)*adjacency.Cols+length+i] += sum / 8
				}
			}
			//result := PageRank(1.0, 33, rng.Uint32(), adjacency)
			result := make([]float64, size)
			graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
				result[node] = rank
			})
			distribution, sum := make([]float64, len(words)), 0.0
			for _, value := range result[:length] {
				if value < 0 {
					value = -value
				}
				sum += float64(value)
			}
			for iii, value := range result[:length] {
				if value < 0 {
					value = -value
				}
				distribution[iii] = float64(value) / sum
			}

			total, selected := 0.0, rng.Float64()
			for iii, value := range distribution {
				total += value
				if selected < total {
					cost += value
					context = append(context, iii)
					break
				}
			}
		}
		done <- Path{
			Path: context,
			Cost: cost,
		}
	}

	rng := rand.New(rand.NewSource(1))
	for range 8 {
		go process(rng.Int63())
	}
	pathes := make([]Path, 8)
	for i := range 8 {
		pathes[i] = <-done
	}
	sort.Slice(pathes, func(i, j int) bool {
		return pathes[i].Cost < pathes[j].Cost
	})
	for _, path := range pathes {
		context := path.Path
		for _, word := range context {
			fmt.Printf("%s ", words[word].Word)
		}
		fmt.Println()
	}
}
