// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"

	"github.com/pointlander/morpheus/kmeans"
)

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
