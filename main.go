// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"flag"
	"fmt"
	"io"
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
}
