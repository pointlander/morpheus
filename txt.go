// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
)

// TxtMode is the txt mode
func TxtMode() {
	const (
		size  = 256
		order = 4
	)

	type File struct {
		Name string
		Data []byte
		Atad *os.File
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
	for i := range files {
		files[i].Data = load(fmt.Sprintf("books/%s", files[i].Name))
	}
	for i := range vectors {
		fmt.Println(i, len(vectors[i]))
	}

	config := Config{
		Iterations: 32,
		Size:       256,
		Divider:    8,
	}
	type T struct{}

	if *FlagLearn {
		done := make(chan bool, 8)
		learn := func(file File, seed int64) {
			rng := rand.New(rand.NewSource(seed))
			output, err := os.Create(fmt.Sprintf("%s.v", file.Name))
			if err != nil {
				panic(err)
			}
			defer output.Close()

			markov := [order]Markov{}
			lines := make([]*Vector[T], 8)
			data := file.Data
			const limit = 8 * 60 * 1024
			if len(data) > limit {
				data = data[:limit]
			}
			for index, value := range data {
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
			done <- true
		}
		rng := rand.New(rand.NewSource(1))
		for _, file := range files {
			go learn(file, rng.Int63())
		}
		for range files {
			<-done
		}
		return
	}

	fmt.Println(*FlagPrompt)
	rng := rand.New(rand.NewSource(1))
	for i := range files {
		input, err := os.Open(fmt.Sprintf("%s.v", files[i].Name))
		if err != nil {
			panic(err)
		}
		files[i].Atad = input
	}
	defer func() {
		for i := range files {
			files[i].Atad.Close()
		}
	}()

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

	//var symbol byte
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

		type Item struct {
			Cosine float32
			Symbol byte
		}
		items := make([]Item, 128)
		//max := float32(0.0)
		for i := range files {
			for {
				vector := NewMatrix[float32](rows*cols, 1)
				err := vector.Read(files[i].Atad)
				if err == io.EOF {
					break
				} else if err != nil {
					panic(err)
				}
				buffer := make([]byte, 1)
				n, err := files[i].Atad.Read(buffer)
				if err != nil {
					panic(err)
				}
				if n != len(buffer) {
					panic(fmt.Errorf("not all bytes read: %d", n))
				}
				cs := mat.CS(vector)
				/*if cs > max {
					max, symbol = cs, buffer[0]
				}*/
				point := 0
				for i := range items {
					if items[i].Cosine < cs {
						point = i
						break
					}
				}
				if point < len(items) {
					cosine := Item{
						Cosine: cs,
						Symbol: buffer[0],
					}
					for i := point; i < len(items); i++ {
						items[i], cosine = cosine, items[i]
					}
				}
			}
		}
		for i := range files {
			files[i].Atad.Seek(0, io.SeekStart)
		}
		sum := float32(0.0)
		for i := range items {
			sum += items[i].Cosine
		}
		total, selected, index := float32(0.0), rng.Float32(), 0
		for i := range items {
			total += items[i].Cosine / sum
			if selected < total {
				index = i
				break
			}
		}
		for i := range markov {
			state := items[index].Symbol
			for ii, value := range markov[i][:i+1] {
				markov[i][ii], state = state, value
			}
		}
		generated = append(generated, items[index].Symbol)
	}
	fmt.Println(string(generated))

}
