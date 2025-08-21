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
	"github.com/pointlander/morpheus/kmeans"
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
	AE       int
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
	const iterations = 512
	results := make([][]float64, iterations)
	for iteration := range iterations {
		a, b := NewMatrix(4, 4, make([]float64, 4*4)...), NewMatrix(4, 4, make([]float64, 4*4)...)
		index := 0
		for range a.Rows {
			for range a.Cols {
				a.Data[index] = rng.NormFloat64()
				b.Data[index] = rng.NormFloat64()
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

	cov := make([][]float64, len(iris))
	for i := range cov {
		cov[i] = make([]float64, len(iris))
	}
	for _, measures := range results {
		for i, v := range measures {
			for ii, vv := range measures {
				diff1 := avg[i] - v
				diff2 := avg[ii] - vv
				cov[i][ii] += diff1 * diff2
			}
		}
	}
	if len(results) > 0 {
		for i := range cov {
			for ii := range cov[i] {
				cov[i][ii] = cov[i][ii] / float64(len(results))
			}
		}
	}
	fmt.Println("K=")
	for i := range cov {
		fmt.Println(cov[i])
	}
	fmt.Println("u=")
	fmt.Println(avg)
	fmt.Println()

	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Index < iris[j].Index
	})

	meta := make([][]float64, len(iris))
	for i := range meta {
		meta[i] = make([]float64, len(iris))
	}
	const k = 3
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
	clusters, _, err := kmeans.Kmeans(1, meta, 3, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, value := range clusters {
		iris[i].Cluster = value
	}
	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Cluster < iris[j].Cluster
	})
	for i := range stddev {
		fmt.Println(iris[i].Cluster, iris[i].Label)
	}
	a := make(map[string][3]int)
	for i := range iris {
		histogram := a[iris[i].Label]
		histogram[iris[i].Cluster]++
		a[iris[i].Label] = histogram
	}
	for k, v := range a {
		fmt.Println(k, v)
	}

	var auto [3]*AutoEncoder
	for i := range auto {
		auto[i] = NewAutoEncoder(len(iris), int64(i+1))
	}
	for iteration := range iterations {
		perm := rng.Perm(len(cov))
		for i := range cov {
			i = perm[i]
			input := make([]float32, len(cov[i]))
			for iii := range cov[i] {
				input[iii] = float32(cov[i][iii])
			}
			if iteration < 32 {
				for ii := range auto {
					auto[ii].Encode(input, input)
				}
				continue
			}
			min, index := float32(math.MaxFloat32), 0
			for ii := range auto {
				e := auto[ii].Measure(input, input)
				if e < min {
					min, index = e, ii
				}
			}
			auto[index].Encode(input, input)
		}
	}
	sort.Slice(iris, func(i, j int) bool {
		return iris[i].Index < iris[j].Index
	})
	for i := range cov {
		input := make([]float32, len(cov[i]))
		for iii := range cov[i] {
			input[iii] = float32(cov[i][iii])
		}
		min, index := float32(math.MaxFloat32), 0
		for ii := range auto {
			e := auto[ii].Measure(input, input)
			if e < min {
				min, index = e, ii
			}
		}
		iris[i].AE = index
	}
	sort.Slice(iris, func(i, j int) bool {
		return iris[i].AE < iris[j].AE
	})
	a = make(map[string][3]int)
	for i := range iris {
		histogram := a[iris[i].Label]
		histogram[iris[i].AE]++
		a[iris[i].Label] = histogram
	}
	for k, v := range a {
		fmt.Println(k, v)
	}
}
