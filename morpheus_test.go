// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"
	"testing"

	"github.com/alixaxel/pagerank"
)

func TestPageRank(t *testing.T) {
	graph := pagerank.NewGraph()

	graph.Link(1, 2, 1.0)
	graph.Link(1, 3, 2.0)
	graph.Link(2, 3, 3.0)
	graph.Link(2, 4, 4.0)
	graph.Link(3, 1, 5.0)

	nodes := make([]float64, 4)
	graph.Rank(0.85, 0.001, func(node uint32, rank float64) {
		nodes[node-1] = rank
	})
	t.Log(nodes)

	adj := NewMatrix(4, 4, make([]float64, 4*4)...)
	adj.Data[0*4+1] = 1.0
	adj.Data[0*4+2] = 2.0
	adj.Data[1*4+2] = 3.0
	adj.Data[1*4+3] = 4.0
	adj.Data[2*4+0] = 5.0
	p := PageRank(.85, 1, adj)
	t.Log(p.Data)
}

func BenchmarkMorpheus(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	config := Config{
		Iterations: 32,
		Size:       256,
		Divider:    8,
	}
	type T struct{}
	vectors := make([]*Vector[T], 8)
	for i := range vectors {
		vector := Vector[T]{}
		for range 256 {
			vector.Vector = append(vector.Vector, rng.Float32())
		}
		vectors[i] = &vector
	}
	for b.Loop() {
		Morpheus(rng.Int63(), config, vectors)
	}
}

func BenchmarkPageRank(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for b.Loop() {
		graph := pagerank.NewGraph()

		for i := range 1024 {
			for j := range 1024 {
				graph.Link(uint32(i), uint32(j), rng.Float64())
			}
		}

		graph.Rank(0.85, 0.001, func(node uint32, rank float64) {

		})
	}
}

func BenchmarkPageRankFast(b *testing.B) {
	rng := rand.New(rand.NewSource(1))
	for b.Loop() {
		adj := NewMatrix(1024, 1024, make([]float64, 1024*1024)...)
		for i := range 1024 {
			for j := range 1024 {
				adj.Data[i*1024+j] = rng.Float64()
			}
		}
		PageRank(.85, 1, adj)
	}
}
