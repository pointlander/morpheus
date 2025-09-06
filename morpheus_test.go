// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math/rand"
	"testing"
)

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
