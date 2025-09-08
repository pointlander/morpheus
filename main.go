// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"os"
	"runtime/pprof"
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

	const (
		size  = 256
		order = 4
	)

	type Markov [order]byte
	var vectors [order]map[Markov][]uint32
	for i := range vectors {
		vectors[i] = make(map[Markov][]uint32)
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
		return data
	}
	load("books/pg74.txt.bz2")
	load("books/10.txt.utf-8.bz2")
	load("books/76.txt.utf-8.bz2")
	load("books/84.txt.utf-8.bz2")
	data := load("books/100.txt.utf-8.bz2")
	load("books/1837.txt.utf-8.bz2")
	load("books/2701.txt.utf-8.bz2")
	load("books/3176.txt.utf-8.bz2")
	for i := range vectors {
		fmt.Println(i, len(vectors[i]))
	}

	rng := rand.New(rand.NewSource(1))
	config := Config{
		Iterations: 32,
		Size:       256,
		Divider:    8,
	}
	type T struct{}

	if *FlagLearn {
		output, err := os.Create("vectors.db")
		if err != nil {
			panic(err)
		}
		defer output.Close()

		markov := [order]Markov{}
		lines := make([]*Vector[T], 8)
		for index, value := range data[:60*1024] {
			line := &Vector[T]{
				Vector: make([]float32, size),
			}
			for i := range markov {
				i = order - 1 - i
				vector := vectors[i][markov[i]]
				if vector != nil {
					for ii := range vector {
						line.Vector[ii] = float32(vector[ii])
					}
					for ii := range lines {
						lines[ii], line = line, lines[ii]
					}
					break
				}
			}
			for i := range markov {
				state := value
				for ii, value := range markov[i][:i+1] {
					markov[i][ii], state = state, value
				}
			}
			for _, line := range lines {
				if line != nil {
					line.Avg = 0
					line.Stddev = 0
				}
			}
			if index < len(lines) {
				continue
			}
			embedding := Morpheus(rng.Int63(), config, lines)
			rows := len(embedding)
			cols := len(embedding[0])
			mat := NewMatrix(rows*cols, 1, make([]float32, rows*cols)...)
			index := 0
			for _, row := range embedding {
				for _, value := range row {
					mat.Data[index] = float32(value)
					index++
				}
			}
			err := mat.Write(output)
			if err != nil {
				panic(err)
			}
			buffer := make([]byte, 1)
			buffer[0] = value
			n, err := output.Write(buffer)
			if err != nil {
				panic(err)
			}
			if n != len(buffer) {
				panic(errors.New("1 bytes should be been written"))
			}
		}
		return
	}

	fmt.Println(*FlagPrompt)
	input, err := os.Open("vectors.db")
	if err != nil {
		panic(err)
	}
	defer input.Close()

	markov := [order]Markov{}
	lines := make([]*Vector[T], 8)
	for _, value := range []byte(*FlagPrompt) {
		line := &Vector[T]{
			Vector: make([]float32, size),
		}
		for i := range markov {
			i = order - 1 - i
			vector := vectors[i][markov[i]]
			if vector != nil {
				for ii := range vector {
					line.Vector[ii] = float32(vector[ii])
				}
				for ii := range lines {
					lines[ii], line = line, lines[ii]
				}
				break
			}
		}
		for i := range markov {
			state := value
			for ii, value := range markov[i][:i+1] {
				markov[i][ii], state = state, value
			}
		}
	}

	var symbol byte
	generated := []byte(*FlagPrompt)
	for range 33 {
		line := &Vector[T]{
			Vector: make([]float32, size),
		}
		for i := range markov {
			i = order - 1 - i
			vector := vectors[i][markov[i]]
			if vector != nil {
				for ii := range vector {
					line.Vector[ii] = float32(vector[ii])
				}
				for ii := range lines {
					lines[ii], line = line, lines[ii]
				}
				break
			}
		}
		for _, line := range lines {
			if line != nil {
				line.Avg = 0
				line.Stddev = 0
			}
		}
		embedding := Morpheus(rng.Int63(), config, lines)
		rows := len(embedding)
		cols := len(embedding[0])
		mat := NewMatrix(rows*cols, 1, make([]float32, rows*cols)...)
		index := 0
		for _, row := range embedding {
			for _, value := range row {
				mat.Data[index] = float32(value)
				index++
			}
		}

		max := float32(0.0)
		for {
			vector := NewMatrix[float32](rows*cols, 1)
			err := vector.Read(input)
			if err == io.EOF {
				break
			} else if err != nil {
				panic(err)
			}
			buffer := make([]byte, 1)
			n, err := input.Read(buffer)
			if err != nil {
				panic(err)
			}
			if n != len(buffer) {
				panic(fmt.Errorf("not all bytes read: %d", n))
			}
			cs := mat.CS(vector)
			if cs > max {
				max, symbol = cs, buffer[0]
			}
		}
		for i := range markov {
			state := symbol
			for ii, value := range markov[i][:i+1] {
				markov[i][ii], state = state, value
			}
		}
		input.Seek(0, io.SeekStart)
		generated = append(generated, symbol)
	}
	fmt.Println(string(generated))
}
