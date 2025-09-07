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
	"runtime"
	"sort"
)

// TextMode is the text generation mode
func TextMode() {
	const (
		iterations = 32
		size       = 256
		n          = 10000
		samples    = 128
		order      = 4
		length     = 16
		segments   = 4
	)

	type Markov [order]byte
	var vectors [order]map[Markov][]uint32
	for i := range vectors {
		vectors[i] = make(map[Markov][]uint32)
	}
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

		markov := [order]Markov{}
		for i, value := range data[:len(data)-6] {
			i += 3
			for ii := range markov {
				vector := vectors[ii][markov[ii]]
				if vector == nil {
					vector = make([]uint32, size)
				}
				vector[data[i-3]]++
				vector[data[i-1]]++
				vector[value]++
				vector[data[i+1]]++
				vector[data[i+3]]++
				vectors[ii][markov[ii]] = vector
				state := value
				for iii, value := range markov[ii][:ii+1] {
					markov[ii][iii], state = state, value
				}
			}
		}
	}
	load("books/pg74.txt.bz2")
	load("books/10.txt.utf-8.bz2")
	load("books/76.txt.utf-8.bz2")
	load("books/84.txt.utf-8.bz2")
	load("books/100.txt.utf-8.bz2")
	load("books/1837.txt.utf-8.bz2")
	load("books/2701.txt.utf-8.bz2")
	load("books/3176.txt.utf-8.bz2")
	for i := range vectors {
		fmt.Println(i, len(vectors[i]))
	}

	type Trace struct {
		Trace string
		Value float64
	}
	vectorize := func(input string, seed int64) Trace {
		rng := rand.New(rand.NewSource(seed))
		type Line struct {
			Symbol byte
		}
		markov := [order]Markov{}
		lines := make([]*Vector[Line], 0, 8)
		for _, value := range []byte(input) {
			line := Vector[Line]{
				Meta: Line{
					Symbol: value,
				},
				Vector: make([]float32, size),
			}
			for i := range markov {
				i = order - 1 - i
				vector := vectors[i][markov[i]]
				if vector != nil {
					for ii := range vector {
						line.Vector[ii] = float32(vector[ii])
					}
					lines = append(lines, &line)
				}
				break
			}
			for i := range markov {
				state := value
				for ii, value := range markov[i][:i+1] {
					markov[i][ii], state = state, value
				}
			}

		}
		count := 0
		for i := range markov {
			i = order - 1 - i
			for ii := range size {
				markov := markov
				state := byte(ii)
				for iv, value := range markov[i][:i+1] {
					markov[i][iv], state = state, value
				}
				vector := vectors[i][markov[i]]
				if vector != nil {
					count++
					line := Vector[Line]{
						Meta: Line{
							Symbol: byte(ii),
						},
						Vector: make([]float32, size),
					}
					for iii, value := range vector {
						line.Vector[iii] = float32(value)
					}
					lines = append(lines, &line)
				}
			}
			if count > 0 {
				break
			}
		}
		//fmt.Println(len(lines), count)

		for i := range lines {
			sum := float32(0.0)
			for _, value := range lines[i].Vector {
				sum += value
			}
			if sum == 0 {
				continue
			}
			for ii, value := range lines[i].Vector {
				lines[i].Vector[ii] = value / sum
			}
		}

		{
			for k := range lines[:len(lines)-count] {
				for i := range size / 2 {
					theta := float64(k) / math.Pow(n, 2*float64(i)/size)
					lines[k].Vector[2*i] += float32(math.Sin(theta))
					lines[k].Vector[2*i+1] += float32(math.Cos(theta))
				}
			}
			k := len(lines) - count
			for kk := len(lines) - count; kk < len(lines); kk++ {
				for i := range size / 2 {
					theta := float64(k) / math.Pow(n, 2*float64(i)/size)
					lines[kk].Vector[2*i] += float32(math.Sin(theta))
					lines[kk].Vector[2*i+1] += float32(math.Cos(theta))
				}
			}
		}

		config := Config{
			Iterations: iterations,
			Size:       size,
			Divider:    8,
		}
		Morpheus(seed, config, lines)

		sum, norm, c := 0.0, make([]float64, count), 0
		for i := len(lines) - count; i < len(lines); i++ {
			sum += lines[i].Stddev
		}
		for i := len(lines) - count; i < len(lines); i++ {
			norm[c] = sum / lines[i].Stddev
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
		next = append(next, lines[(len(lines)-count)+index].Meta.Symbol)

		sum = 0.0
		for i := 0; i < len(lines)-count; i++ {
			sum += lines[i].Avg
		}
		sum += lines[(len(lines)-count)+index].Avg

		return Trace{
			Trace: string(next),
			Value: lines[(len(lines)-count)+index].Avg / sum,
		}
	}

	done := make(chan Trace, 8)
	trace := func(input string, seed int64) {
		rng := rand.New(rand.NewSource(seed))
		t := Trace{
			Trace: input,
		}
		for range length {
			trace := vectorize(t.Trace, rng.Int63())
			t.Trace = trace.Trace
			t.Value += trace.Value
		}
		done <- t
	}

	rng := rand.New(rand.NewSource(1))
	state := "What is the meaning of life?"
	for range segments {
		traces := make([]Trace, 0, samples)
		index, flight, cpus := 0, 0, runtime.NumCPU()
		for index < samples && flight < cpus {
			go trace(state, rng.Int63())
			index++
			flight++
		}
		for index < samples {
			traces = append(traces, <-done)
			flight--

			go trace(state, rng.Int63())
			index++
			flight++
		}
		for range flight {
			traces = append(traces, <-done)
		}
		sort.Slice(traces, func(i, j int) bool {
			return traces[i].Value > traces[j].Value
		})
		for _, t := range traces {
			fmt.Println(t.Value)
			fmt.Println(t.Trace)
			fmt.Println("-----------------------")
		}
		state = traces[0].Trace
	}
	fmt.Println()
	fmt.Println(state)

}
