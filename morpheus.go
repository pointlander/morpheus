// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"math/rand"

	"github.com/alixaxel/pagerank"
)

// Vector is a vector
type Vector[T any] struct {
	Meta   T
	Vector []float32
	Avg    float64
	Stddev float64
}

// Config is the configuration for the morpheus algorithm
type Config struct {
	Iterations int
	Size       int
	Divider    int
}

func Morpheus[T any](seed int64, config Config, vectors []*Vector[T]) [][]float64 {
	rng := rand.New(rand.NewSource(seed))
	results := make([][]float64, config.Iterations)
	width := 2 * config.Size
	for iteration := range config.Iterations {
		a, b := NewMatrix(width, width/config.Divider, make([]float32, width*width/config.Divider)...),
			NewMatrix(width, width/config.Divider, make([]float32, width*width/config.Divider)...)
		index := 0
		for range a.Rows {
			for range a.Cols {
				a.Data[index] = float32(rng.NormFloat64())
				b.Data[index] = float32(rng.NormFloat64())
				index++
			}
		}
		aa := a.Softmax(1)
		bb := b.Softmax(1)
		graph := pagerank.NewGraph()
		for i := range vectors {
			x := NewMatrix(width, 1, make([]float32, width)...)
			for ii, value := range vectors[i].Vector {
				if value < 0 {
					x.Data[config.Size+ii] = -value
					continue
				}
				x.Data[ii] = value
			}
			xx := aa.MulT(x)
			for ii := range vectors {
				y := NewMatrix(width, 1, make([]float32, width)...)
				for iii, value := range vectors[ii].Vector {
					if value < 0 {
						y.Data[config.Size+iii] = -value
						continue
					}
					y.Data[iii] = value
				}
				yy := bb.MulT(y)
				cs := xx.CS(yy)
				graph.Link(uint32(i), uint32(ii), float64(cs))
			}
		}
		result := make([]float64, len(vectors))
		graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
			result[node] = rank
		})
		results[iteration] = result
	}
	for _, result := range results {
		for i, value := range result {
			vectors[i].Avg += value
		}
	}
	for i, value := range vectors {
		vectors[i].Avg = value.Avg / float64(config.Iterations)
	}

	for _, result := range results {
		for i, value := range result {
			diff := value - vectors[i].Avg
			vectors[i].Stddev += diff * diff
		}
	}
	for i, value := range vectors {
		vectors[i].Stddev = math.Sqrt(value.Stddev / float64(config.Iterations))
	}

	cov := make([][]float64, len(vectors))
	for i := range cov {
		cov[i] = make([]float64, len(vectors))
	}
	for _, measures := range results {
		for i, v := range measures {
			for ii, vv := range measures {
				diff1 := vectors[i].Avg - v
				diff2 := vectors[ii].Avg - vv
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
	return cov
}
