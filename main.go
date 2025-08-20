// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"archive/zip"
	"bytes"
	"embed"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"
	"strconv"

	"github.com/alixaxel/pagerank"
)

//go:embed iris.zip
var Iris embed.FS

// Fisher is the fisher iris data
type Fisher struct {
	Measures []float64
	Label    string
	Cluster  int
	Index    int
	Rank     float64
}

// Labels maps iris labels to ints
var Labels = map[string]int{
	"Iris-setosa":     0,
	"Iris-versicolor": 1,
	"Iris-virginica":  2,
}

// Inverse is the labels inverse map
var Inverse = [3]string{
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica",
}

// Load loads the iris data set
func Load() []Fisher {
	file, err := Iris.Open("iris.zip")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	fisher := make([]Fisher, 0, 8)
	reader, err := zip.NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		panic(err)
	}
	for _, f := range reader.File {
		if f.Name == "iris.data" {
			iris, err := f.Open()
			if err != nil {
				panic(err)
			}
			reader := csv.NewReader(iris)
			data, err := reader.ReadAll()
			if err != nil {
				panic(err)
			}
			for i, item := range data {
				record := Fisher{
					Measures: make([]float64, 4),
					Label:    item[4],
					Index:    i,
				}
				for ii := range item[:4] {
					f, err := strconv.ParseFloat(item[ii], 64)
					if err != nil {
						panic(err)
					}
					record.Measures[ii] = f
				}
				fisher = append(fisher, record)
			}
			iris.Close()
		}
	}
	return fisher
}

func main() {
	iris := Load()
	rng := rand.New(rand.NewSource(1))
	const iterations = 128
	results := make([][]float64, iterations)
	for iteration := range iterations {
		a, b := NewMatrix(4, 4, make([]float64, 4*4)...), NewMatrix(4, 4, make([]float64, 4*4)...)
		index := 0
		for range a.Rows {
			for range a.Cols {
				a.Data[index] = rng.Float64()
				b.Data[index] = rng.Float64()
				index++
			}
		}
		a = a.Softmax(1)
		b = b.Softmax(1)
		graph := pagerank.NewGraph()
		for i := range iris {
			for ii := range iris {
				x, y := NewMatrix(4, 1, make([]float64, 4)...), NewMatrix(4, 1, make([]float64, 4)...)
				for i, value := range iris[i].Measures {
					x.Data[i] = value
				}
				for i, value := range iris[ii].Measures {
					y.Data[i] = value
				}
				x = a.MulT(x)
				y = b.MulT(y)
				cs := x.CS(y)
				graph.Link(uint32(i), uint32(ii), cs)
			}
		}
		result := make([]float64, len(iris))
		graph.Rank(1.0, 1e-6, func(node uint32, rank float64) {
			result[node] = rank
		})
		results[iteration] = result
	}
	avg := make([]float64, len(iris))
	for _, result := range results {
		for i, value := range result {
			avg[i] += value
		}
	}
	for i, value := range avg {
		avg[i] = value / float64(iterations)
	}
	stddev := make([]float64, len(iris))
	for _, result := range results {
		for i, value := range result {
			diff := value - avg[i]
			stddev[i] += diff * diff
		}
	}
	for i, value := range stddev {
		stddev[i] = math.Sqrt(value / float64(iterations))
	}
	for i := range iris {
		iris[i].Rank = stddev[i]
	}
	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Rank < iris[j].Rank
	})
	for i := range stddev {
		fmt.Println(iris[i].Label)
	}
}
