// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"fmt"
	"io"
	"math/rand"
)

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
			book.Model[i] = make(map[Markov][]float32)
		}
		for _, value := range data {
			for ii := range markov {
				vector := book.Model[ii][markov[ii]]
				if vector == nil {
					vector = make([]float32, size)
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
