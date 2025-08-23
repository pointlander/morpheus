// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf64"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

// AutoEncoder is an autoencoder
type AutoEncoder struct {
	Set       tf64.Set
	Rng       *rand.Rand
	Iteration int
}

// NewAutoEncoder creates a new autoencoder
func NewAutoEncoder(size int, seed int64) *AutoEncoder {
	a := AutoEncoder{
		Rng: rand.New(rand.NewSource(seed)),
	}
	set := tf64.NewSet()
	set.Add("l1", size, size/6)
	set.Add("b1", size/6, 1)
	set.Add("l2", size/3, size)
	set.Add("b2", size, 1)

	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float64, StateTotal)
			for ii := range w.States {
				w.States[ii] = make([]float64, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for range cap(w.X) {
			w.X = append(w.X, a.Rng.NormFloat64()*factor)
		}
		w.States = make([][]float64, StateTotal)
		for ii := range w.States {
			w.States[ii] = make([]float64, len(w.X))
		}
	}

	a.Set = set
	return &a
}

// Measure measures the loss of a single input
func (a *AutoEncoder) Measure(input, output []float64) float64 {
	others := tf64.NewSet()
	size := len(input)
	others.Add("input", size, 1)
	others.Add("output", size, 1)
	in := others.ByName["input"]
	in.X = append(in.X, input...)
	out := others.ByName["output"]
	out.X = append(out.X, output...)
	l1 := tf64.Everett(tf64.Add(tf64.Mul(a.Set.Get("l1"), others.Get("input")), a.Set.Get("b1")))
	l2 := tf64.Add(tf64.Mul(a.Set.Get("l2"), l1), a.Set.Get("b2"))
	loss := tf64.Sum(tf64.Quadratic(l2, others.Get("output")))

	l := 0.0
	loss(func(a *tf64.V) bool {
		l = a.X[0]
		return true
	})
	return l
}

func (a *AutoEncoder) pow(x float64) float64 {
	y := math.Pow(x, float64(a.Iteration+1))
	if math.IsNaN(y) || math.IsInf(y, 0) {
		return 0
	}
	return y
}

// Encode encodes a single input
func (a *AutoEncoder) Encode(input, output []float64) float64 {
	rng := rand.New(rand.NewSource(a.Rng.Int63()))
	others := tf64.NewSet()
	size := len(input)
	others.Add("input", size, 1)
	others.Add("output", size, 1)
	in := others.ByName["input"]
	in.X = append(in.X, input...)
	out := others.ByName["output"]
	out.X = append(out.X, output...)

	dropout := map[string]interface{}{
		"rng": rng,
	}

	l1 := tf64.Dropout(tf64.Everett(tf64.Add(tf64.Mul(a.Set.Get("l1"), others.Get("input")), a.Set.Get("b1"))), dropout)
	l2 := tf64.Add(tf64.Mul(a.Set.Get("l2"), l1), a.Set.Get("b2"))
	loss := tf64.Avg(tf64.Quadratic(l2, others.Get("output")))

	l := 0.0
	a.Set.Zero()
	others.Zero()
	l = tf64.Gradient(loss).X[0]
	if math.IsNaN(float64(l)) || math.IsInf(float64(l), 0) {
		fmt.Println(a.Iteration, l)
		return 0
	}

	norm := 0.0
	for _, p := range a.Set.Weights {
		for _, d := range p.D {
			norm += d * d
		}
	}
	norm = math.Sqrt(norm)
	b1, b2 := a.pow(B1), a.pow(B2)
	scaling := 1.0
	if norm > 1 {
		scaling = 1 / norm
	}
	for _, w := range a.Set.Weights {
		for ii, d := range w.D {
			g := d * scaling
			m := B1*w.States[StateM][ii] + (1-B1)*g
			v := B2*w.States[StateV][ii] + (1-B2)*g*g
			w.States[StateM][ii] = m
			w.States[StateV][ii] = v
			mhat := m / (1 - b1)
			vhat := v / (1 - b2)
			if vhat < 0 {
				vhat = 0
			}
			w.X[ii] -= Eta * mhat / (math.Sqrt(vhat) + 1e-8)
		}
	}
	a.Iteration++
	return l
}
