// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"runtime/pprof"
	"sort"

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
	// FlagLearn learn the vector database
	FlagLearn = flag.Bool("learn", false, "learn the vector database")
	// FlagPrompt the prompt to use
	FlagPrompt = flag.String("prompt", "What is the meaning of life?", "the prompt to use")
	// cpuprofile profiles the program
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to `file`")
)

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

	const (
		size  = 256
		width = 33
		order = 4
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

	type Markov [order]byte
	var A, B [order]map[Markov][]uint32
	for i := range A {
		A[i] = make(map[Markov][]uint32)
	}
	for i := range B {
		B[i] = make(map[Markov][]uint32)
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
				vector := A[ii][markov[ii]]
				if vector == nil {
					vector = make([]uint32, size)
				}
				vector[value]++
				A[ii][markov[ii]] = vector

				vector = B[ii][markov[ii]]
				if vector == nil {
					vector = make([]uint32, size)
				}
				vector[value]++
				B[ii][markov[ii]] = vector

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
	for i := range A {
		fmt.Println(i, len(A[i]))
	}

	type Segment struct {
		Segment []byte
		Rank    float64
		Cluster int
	}

	rng := rand.New(rand.NewSource(1))
	for range 3 {
		segments := []*Vector[Segment]{}
		input := []byte(*FlagPrompt)
		length := len(input) + width
		sets := [][order]map[Markov][]uint32{A, B}
		for _, vectors := range sets {
			for range 1024 {
				segment := Vector[Segment]{}
				markov := [order]Markov{}
				var val byte
				for _, val = range input {
					for i := range markov {
						i = order - 1 - i
						vector := vectors[i][markov[i]]
						if vector != nil {
							sum := float32(0.0)
							for _, value := range vector {
								sum += float32(value)
							}
							segment.Meta.Segment = append(segment.Meta.Segment, val)
							segment.Vector = append(segment.Vector, float32(vector[val])/sum)
							break
						}
					}
					for i := range markov {
						state := val
						for ii, value := range markov[i][:i+1] {
							markov[i][ii], state = state, value
						}
					}
				}

				for range width {
					for i := range markov {
						i = order - 1 - i
						vector := vectors[i][markov[i]]
						if vector != nil {
							sum := float32(0.0)
							for _, value := range vector {
								sum += float32(value)
							}
							total, selection := float32(0.0), rng.Float32()
							for i, value := range vector {
								total += float32(value) / sum
								if selection < total {
									segment.Meta.Segment = append(segment.Meta.Segment, byte(i))
									val = byte(i)
									segment.Vector = append(segment.Vector, float32(value)/sum)
									break
								}
							}
							break
						}
					}
					for i := range markov {
						state := val
						for ii, value := range markov[i][:i+1] {
							markov[i][ii], state = state, value
						}
					}
				}
				segments = append(segments, &segment)
			}
		}

		config := Config{
			Iterations: 8,
			Size:       length,
			Divider:    1,
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

		for i := range A {
			A[i] = make(map[Markov][]uint32)
		}
		for i := range B {
			B[i] = make(map[Markov][]uint32)
		}
		for i := range segments {
			markov := [order]Markov{}
			vectors := A
			if segments[i].Meta.Cluster == 1 {
				vectors = B
			}
			for _, value := range segments[i].Meta.Segment {
				for iii := range markov {
					vector := vectors[iii][markov[iii]]
					if vector == nil {
						vector = make([]uint32, size)
					}
					vector[value]++
					vectors[iii][markov[iii]] = vector
					state := value
					for iv, value := range markov[iii][:iii+1] {
						markov[iii][iv], state = state, value
					}
				}
			}
		}
	}
}
