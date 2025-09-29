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

// IrisMode is the iris clustering mode
func IrisMode() {
	rng := rand.New(rand.NewSource(1))
	vectors := make([]*Vector[Fisher], 150)
	{
		iris := Load()
		for i := range vectors {
			vector := Vector[Fisher]{}
			vector.Meta = iris[i]
			for _, value := range iris[i].Measures {
				vector.Vector = append(vector.Vector, float32(value))
			}
			vectors[i] = &vector
		}
	}
	config := Config{
		Iterations: 1024,
		Size:       4,
		Divider:    1,
	}
	cov := MorpheusGramSchmidt(rng.Int63(), config, vectors)

	sort.Slice(vectors, func(i, j int) bool {
		return vectors[i].Stddev < vectors[j].Stddev
	})
	for i := range vectors {
		fmt.Println(vectors[i].Meta.Label)
	}

	fmt.Println("K=")
	for i := range cov {
		fmt.Println(cov[i])
	}
	fmt.Println("u=")
	for i := range vectors {
		fmt.Printf("%v ", vectors[i].Avg)
	}
	fmt.Println()

	sort.Slice(vectors, func(i, j int) bool {
		return vectors[i].Meta.Index < vectors[j].Meta.Index
	})

	meta := make([][]float64, len(vectors))
	for i := range meta {
		meta[i] = make([]float64, len(vectors))
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
		vectors[i].Meta.Cluster = value
	}
	sort.Slice(vectors, func(i, j int) bool {
		return vectors[i].Meta.Cluster < vectors[j].Meta.Cluster
	})
	for i := range vectors {
		fmt.Println(vectors[i].Meta.Cluster, vectors[i].Meta.Label)
	}
	a := make(map[string][3]int)
	for i := range vectors {
		histogram := a[vectors[i].Meta.Label]
		histogram[vectors[i].Meta.Cluster]++
		a[vectors[i].Meta.Label] = histogram
	}
	for k, v := range a {
		fmt.Println(k, v)
	}

	if !*FlagE {
		return
	}

	var auto [3]*AutoEncoder
	for i := range auto {
		auto[i] = NewAutoEncoder(len(vectors), 1)
	}
	for i := range cov {
		sum := 0.0
		for _, value := range cov[i] {
			sum += value
		}
		for ii, value := range cov[i] {
			cov[i][ii] = value / sum
		}
	}
	for range 32 {
		var histogram [3]int
		for i := range cov {
			min, minIndex := math.MaxFloat32, 0
			for ii := range auto {
				e := auto[ii].Measure(cov[i], cov[i])
				if e < min {
					min, minIndex = e, ii
				}
			}
			histogram[minIndex]++
		}
		fmt.Println(histogram)
		min, minIndex := math.MaxInt64, 0
		for i, value := range histogram {
			if value < min {
				min, minIndex = value, i
			}
		}
		perm := rng.Perm(len(cov))
		for i := range cov {
			i = perm[i]
			auto[minIndex].Encode(cov[i], cov[i])
		}
	}
}

// IrisMarkovMode is the iris markov clustering mode
func IrisMarkovMode() {
	rng := rand.New(rand.NewSource(1))
	vectors := make([]*Vector[Fisher], 150)
	{
		iris := Load()
		for i := range vectors {
			vector := Vector[Fisher]{}
			vector.Meta = iris[i]
			for _, value := range iris[i].Measures {
				vector.Vector = append(vector.Vector, float32(value))
			}
			vectors[i] = &vector
		}
	}
	config := Config{
		Iterations: 512,
		Size:       4,
		Divider:    1,
	}
	markov := MorpheusMarkov[Fisher, float64](rng.Int63(), config, vectors)
	cov := make([][]float64, markov.Rows)
	fmt.Println("K=")
	for i := range markov.Rows {
		cov[i] = markov.Data[i*markov.Cols : (i+1)*markov.Cols]
		fmt.Println(cov[i])
	}
	fmt.Println()

	sort.Slice(vectors, func(i, j int) bool {
		return vectors[i].Meta.Index < vectors[j].Meta.Index
	})

	meta := make([][]float64, len(vectors))
	for i := range meta {
		meta[i] = make([]float64, len(vectors))
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
		vectors[i].Meta.Cluster = value
	}
	sort.Slice(vectors, func(i, j int) bool {
		return vectors[i].Meta.Cluster < vectors[j].Meta.Cluster
	})
	for i := range vectors {
		fmt.Println(vectors[i].Meta.Cluster, vectors[i].Meta.Label)
	}
	a := make(map[string][3]int)
	for i := range vectors {
		histogram := a[vectors[i].Meta.Label]
		histogram[vectors[i].Meta.Cluster]++
		a[vectors[i].Meta.Label] = histogram
	}
	for k, v := range a {
		fmt.Println(k, v)
	}

	if !*FlagE {
		return
	}

	var auto [3]*AutoEncoder
	for i := range auto {
		auto[i] = NewAutoEncoder(len(vectors), 1)
	}
	for i := range cov {
		sum := 0.0
		for _, value := range cov[i] {
			sum += value
		}
		for ii, value := range cov[i] {
			cov[i][ii] = value / sum
		}
	}
	for range 32 {
		var histogram [3]int
		for i := range cov {
			min, minIndex := math.MaxFloat32, 0
			for ii := range auto {
				e := auto[ii].Measure(cov[i], cov[i])
				if e < min {
					min, minIndex = e, ii
				}
			}
			histogram[minIndex]++
		}
		fmt.Println(histogram)
		min, minIndex := math.MaxInt64, 0
		for i, value := range histogram {
			if value < min {
				min, minIndex = value, i
			}
		}
		perm := rng.Perm(len(cov))
		for i := range cov {
			i = perm[i]
			auto[minIndex].Encode(cov[i], cov[i])
		}
	}
}
