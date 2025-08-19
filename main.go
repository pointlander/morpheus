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
	results := make([][]float64, 0, 128)
	for range 128 {
		a, b := NewMatrix[float64](4, 4), NewMatrix[float64](4, 4)
		for range a.Rows {
			for range a.Cols {
				a.Data = append(a.Data, rng.Float64())
				b.Data = append(b.Data, rng.Float64())
			}
		}
		a = a.Softmax(1)
		b = b.Softmax(1)
		graph := pagerank.NewGraph()
		for i := range iris {
			for ii := range iris {
				x, y := NewMatrix[float64](4, 1), NewMatrix[float64](4, 1)
				for _, value := range iris[i].Measures {
					x.Data = append(x.Data, value)
				}
				for _, value := range iris[ii].Measures {
					y.Data = append(y.Data, value)
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
		results = append(results, result)
	}
	avg := make([]float64, len(iris))
	for _, result := range results {
		for i, value := range result {
			avg[i] += value
		}
	}
	for i, value := range avg {
		avg[i] = value / 128.0
	}
	stddev := make([]float64, len(iris))
	for _, result := range results {
		for i, value := range result {
			diff := value - avg[i]
			stddev[i] += diff * diff
		}
	}
	for i, value := range stddev {
		stddev[i] = math.Sqrt(value / 128.0)
	}
	for i := range stddev {
		fmt.Println(stddev[i], iris[i].Label)
	}
}
