// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bufio"
	"compress/bzip2"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/pointlander/morpheus/kmeans"
)

func Mach1Mode() {
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

	type Line struct {
		Cluster int
		Count   int
	}

	bible := string(files[1].Data)
	reg := regexp.MustCompile(`\s+`)
	parts := reg.Split(bible, -1)
	reg = regexp.MustCompile(`[\p{P}]+`)
	unique := make(map[string]*Vector[Line])
	links := make(map[string]map[string]uint64)
	previous, current := "", ""
	for _, part := range parts {
		parts := reg.Split(part, -1)
		parts = append(parts, reg.FindAllString(part, -1)...)
		for _, part := range parts {
			_, err := strconv.Atoi(part)
			if err == nil {
				continue
			}
			word := strings.ToLower(strings.TrimSpace(part))
			count := unique[word]
			if count == nil {
				count = &Vector[Line]{
					Word: word,
					Meta: Line{},
				}
			}
			count.Meta.Count++
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
	words := make([]*Vector[Line], len(unique))
	{
		count := 0
		for _, word := range unique {
			words[count] = word
			count++
		}
	}
	sort.Slice(words, func(i, j int) bool {
		if words[i].Meta.Count == words[j].Meta.Count {
			return words[i].Word < words[j].Word
		}
		return words[i].Meta.Count > words[j].Meta.Count
	})
	fmt.Println(len(words))

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

	index := make(map[string]*Vector[Line])
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, " ")
		word := Vector[Line]{
			Word:   strings.TrimSpace(parts[0]),
			Meta:   Line{},
			Vector: make([]float32, 50),
		}
		for i, part := range parts[1:] {
			value, err := strconv.ParseFloat(strings.TrimSpace(part), 32)
			if err != nil {
				panic(err)
			}
			word.Vector[i] = float32(value)
		}
		index[word.Word] = &word
	}

	for i := range words {
		vector := index[words[i].Word]
		if vector != nil {
			words[i].Vector = vector.Vector
		}
	}
	clean := make([]*Vector[Line], 0, 8)
	for i := range words {
		if words[i].Vector != nil {
			clean = append(clean, words[i])
		}
	}
	words = clean

	config := Config{
		Iterations: 1,
		Size:       50,
		Divider:    1,
		Accuracy:   8,
	}
	words = words[:1024]
	{
		rng := rand.New(rand.NewSource(1))
		type Trace struct {
			Trace []*Vector[Line]
			Value float64
		}
		traces := make([]Trace, 7)
		context := "lord"
		for i := range traces {
			fmt.Println("trace", i)
			state, index := context, 0
			for ii := range words {
				if words[ii].Word == state {
					index = ii
					break
				}
			}

			indexes := []int{index}
			const weight = 256
			for ii := range 33 {
				fmt.Println("word", ii)
				MorpheusGramSchmidt(rng.Int63(), config, words, func(cs *Matrix[float32]) {
					fmt.Println("cs")
					for i := range words {
						from := links[words[i].Word]
						for ii := range words {
							to := from[words[ii].Word]
							if to == 0 {
								cs.Data[i*cs.Cols+ii] = 0
							}
						}
					}
					for c, col := range indexes {
					loop:
						for i := range cs.Rows {
							if c > 0 && indexes[c-1] == i {
								cs.Data[i*cs.Cols+col] = weight
							} else {
								for _, value := range indexes {
									if value == i && col != i {
										continue loop
									}
								}
								cs.Data[i*cs.Cols+col] = weight
							}
						}
					}
				})
				distribution, sum := make([]float64, len(words)), 0.0
				for _, value := range words {
					stddev := value.Avg
					if stddev < 0 {
						stddev = -stddev
					}
					sum += stddev
				}
				for iii, value := range words {
					stddev := value.Avg
					if stddev < 0 {
						stddev = -stddev
					}
					distribution[iii] = stddev / sum
				}

				total, selected := 0.0, rng.Float64()
			sampling:
				for iii, value := range distribution {
					total += value
					for _, value := range indexes {
						if value == iii {
							continue sampling
						}
					}
					if selected < total {
						index = iii
						indexes = append(indexes, index)
						traces[i].Trace = append(traces[i].Trace, words[index])
						fmt.Println(words[index].Word)
						state = words[index].Word
						traces[i].Value += math.Abs(words[index].Avg)
						break
					}
				}
			}
		}

		sort.Slice(traces, func(i, j int) bool {
			return traces[i].Value > traces[j].Value
		})
		for i := range traces {
			for _, word := range traces[i].Trace {
				fmt.Printf("%s ", word.Word)
			}
			fmt.Println()
		}
	}

	cov := Morpheus(1, config, words)
	meta := make([][]float64, len(words))
	for i := range meta {
		meta[i] = make([]float64, len(words))
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
	for i := range words {
		words[i].Meta.Cluster = clusters[i]
	}
	sort.Slice(words, func(i, j int) bool {
		return words[i].Stddev < words[j].Stddev
	})

	output, err := os.Create("report.html")
	if err != nil {
		panic(err)
	}
	defer output.Close()
	fmt.Fprintln(output, `<style>
	  table, th, td {
	    border: 1px solid black;
	    border-collapse: collapse; /* Collapses borders into a single border */
	  }
	  th, td {
	    padding: 8px; /* Add padding to cells for better readability */
	    text-align: left;
	  }
	</style>`)
	fmt.Fprintf(output, "<table>\n")
	fmt.Fprintf(output, " <tr>\n")
	fmt.Fprintf(output, "  <th>Word</th>\n")
	fmt.Fprintf(output, "  <th>Count</th>\n")
	fmt.Fprintf(output, "  <th>Cluster</th>\n")
	fmt.Fprintf(output, "  <th>Standard Deviation</th>\n")
	fmt.Fprintf(output, " </tr>\n")
	for i := range words {
		fmt.Fprintf(output, " <tr>\n")
		fmt.Fprintf(output, "  <td>%s</td>\n", words[i].Word)
		fmt.Fprintf(output, "  <td>%d</td>\n", words[i].Meta.Count)
		fmt.Fprintf(output, "  <td>%d</td>\n", words[i].Meta.Cluster)
		fmt.Fprintf(output, "  <td>%.8f</td>\n", words[i].Stddev)
		fmt.Fprintf(output, " </tr>\n")
	}
	fmt.Fprintf(output, "</table>\n")

}
