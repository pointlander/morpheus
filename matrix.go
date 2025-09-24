// Copyright 2024 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"sync/atomic"

	"github.com/pointlander/morpheus/vector"
)

const (
	// S is the scaling factor for the softmax
	S = 1.0 - 1e-300
)

// Number is a number
type Float interface {
	float32 | float64
}

// Size is the size of the matrix
type Size struct {
	Name string
	Cols int
	Rows int
}

// Set is a set of sizes
type Set[T Float] struct {
	Sizes   []Size
	ByIndex []Matrix[T]
	ByName  map[string]*Matrix[T]
}

// Matrix is a float64 matrix
type Matrix[T Float] struct {
	Size
	Data []T
}

// NewMatrix creates a new float64 matrix
func NewMatrix[T Float](cols, rows int, data ...T) Matrix[T] {
	if data == nil {
		data = make([]T, 0, cols*rows)
	}
	return Matrix[T]{
		Size: Size{
			Cols: cols,
			Rows: rows,
		},
		Data: data,
	}
}

// NewMatrices creates matrices from a set
func NewMatrices[T Float](set Set[T], weights []T) Set[T] {
	offset := 0
	set.ByIndex = make([]Matrix[T], len(set.Sizes))
	set.ByName = make(map[string]*Matrix[T])
	for i, size := range set.Sizes {
		end := size.Cols * size.Rows
		set.ByIndex[i] = NewMatrix(size.Cols, size.Rows, weights[offset:offset+end]...)
		factor := math.Sqrt(2.0 / float64(size.Cols))
		for ii := range set.ByIndex[i].Data {
			set.ByIndex[i].Data[ii] *= T(factor)
		}
		set.ByName[size.Name] = &set.ByIndex[i]
		offset += end
	}
	return set
}

// Size is the size of the weight vector
func (s Set[T]) Size() int {
	sum := 0
	for _, size := range s.Sizes {
		sum += size.Cols * size.Rows
	}
	return sum
}

// Named returns the matrix named name
func (s Set[T]) Named(name string) Matrix[T] {
	return *s.ByName[name]
}

// MulT multiplies two matrices and computes the transpose
func (m Matrix[T]) MulT(n Matrix[T]) Matrix[T] {
	if m.Cols != n.Cols {
		panic(fmt.Errorf("%d != %d", m.Cols, n.Cols))
	}
	columns := m.Cols
	o := Matrix[T]{
		Size: Size{
			Cols: m.Rows,
			Rows: n.Rows,
		},
		Data: make([]T, 0, m.Rows*n.Rows),
	}
	lenn, lenm := len(n.Data), len(m.Data)
	for i := 0; i < lenn; i += columns {
		nn := n.Data[i : i+columns]
		for j := 0; j < lenm; j += columns {
			mm := m.Data[j : j+columns]
			o.Data = append(o.Data, dot(mm, nn))
		}
	}
	return o
}

// Add adds two float64 matrices
func (m Matrix[T]) Add(n Matrix[T]) Matrix[T] {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: m.Rows,
		},
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value+n.Data[i%lenb])
	}
	return o
}

// Sub subtracts two float64 matrices
func (m Matrix[T]) Sub(n Matrix[T]) Matrix[T] {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: m.Rows,
		},
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value-n.Data[i%lenb])
	}
	return o
}

// Hadamard multiples two float64 matrices
func (m Matrix[T]) Hadamard(n Matrix[T]) Matrix[T] {
	lena, lenb := len(m.Data), len(n.Data)
	if lena%lenb != 0 {
		panic(fmt.Errorf("%d %% %d != 0", lena, lenb))
	}

	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: m.Rows,
		},
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i, value := range m.Data {
		o.Data = append(o.Data, value*n.Data[i%lenb])
	}
	return o
}

// Softmax calculates the softmax of the matrix rows
func (m Matrix[T]) Softmax(t T) Matrix[T] {
	output := NewMatrix[T](m.Cols, m.Rows)
	max := T(0.0)
	for _, v := range m.Data {
		v /= t
		if v > max {
			max = v
		}
	}
	s := max * S
	for i := 0; i < len(m.Data); i += m.Cols {
		sum := T(0.0)
		values := make([]T, m.Cols)
		for j, value := range m.Data[i : i+m.Cols] {
			values[j] = T(math.Exp(float64(value/t - s)))
			sum += values[j]
		}
		for _, value := range values {
			output.Data = append(output.Data, value/sum)
		}
	}
	return output
}

// Sigmoid computes the sigmoid of a matrix
func (m Matrix[T]) Sigmoid() Matrix[T] {
	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: m.Rows,
		},
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		o.Data = append(o.Data, 1/(1+T(math.Exp(float64(-value)))))
	}
	return o
}

// ReLu is the ramp function
func (m Matrix[T]) ReLu() Matrix[T] {
	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: m.Rows,
		},
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for _, value := range m.Data {
		if value < 0 {
			value = 0
		}
		o.Data = append(o.Data, value)
	}
	return o
}

// Entropy calculates the entropy of the matrix rows
func (m Matrix[T]) Entropy() Matrix[T] {
	output := NewMatrix[T](m.Rows, 1)
	for i := 0; i < len(m.Data); i += m.Cols {
		entropy := T(0.0)
		for _, value := range m.Data[i : i+m.Cols] {
			entropy += value * T(math.Log(float64(value)))
		}
		output.Data = append(output.Data, -entropy)
	}
	return output
}

// Sum sums the rows of a matrix
func (m Matrix[T]) Sum() Matrix[T] {
	o := Matrix[T]{
		Size: Size{
			Cols: m.Cols,
			Rows: 1,
		},
		Data: make([]T, m.Cols),
	}
	for i := 0; i < m.Rows; i++ {
		offset := i * m.Cols
		for j := range o.Data {
			o.Data[j] += m.Data[offset+j]
		}
	}
	return o
}

// T tramsposes a matrix
func (m Matrix[T]) T() Matrix[T] {
	o := Matrix[T]{
		Size: Size{
			Cols: m.Rows,
			Rows: m.Cols,
		},
		Data: make([]T, 0, m.Cols*m.Rows),
	}
	for i := 0; i < m.Cols; i++ {
		for j := 0; j < m.Rows; j++ {
			o.Data = append(o.Data, m.Data[j*m.Cols+i])
		}
	}
	return o
}

// Write writes the matrix to a file
func (m Matrix[T]) Write(output *os.File) error {
	switch data := any(m.Data).(type) {
	case []float64:
		buffer64 := make([]byte, 8)
		for _, parameter := range data {
			bits := math.Float64bits(parameter)
			for i := range buffer64 {
				buffer64[i] = byte((bits >> (8 * i)) & 0xFF)
			}
			n, err := output.Write(buffer64)
			if err != nil {
				return err
			}
			if n != len(buffer64) {
				return errors.New("8 bytes should be been written")
			}
		}
	case []float32:
		buffer32 := make([]byte, 4)
		for _, parameter := range data {
			bits := math.Float32bits(parameter)
			for i := range buffer32 {
				buffer32[i] = byte((bits >> (8 * i)) & 0xFF)
			}
			n, err := output.Write(buffer32)
			if err != nil {
				return err
			}
			if n != len(buffer32) {
				return errors.New("8 bytes should be been written")
			}
		}
	}
	return nil
}

// Read reads the matrix from a file
func (m *Matrix[T]) Read(input *os.File) error {
	switch any(m.Data).(type) {
	case []float64:
		buffer64 := make([]byte, 8)
		for range m.Rows {
			for range m.Cols {
				n, err := input.Read(buffer64)
				if err == io.EOF {
					return err
				} else if err != nil {
					return err
				}
				if n != len(buffer64) {
					return fmt.Errorf("not all bytes read: %d", n)
				}
				value := uint64(0)
				for k := 0; k < 8; k++ {
					value <<= 8
					value |= uint64(buffer64[7-k])
				}
				m.Data = append(m.Data, T(math.Float64frombits(value)))
			}
		}
	case []float32:
		buffer32 := make([]byte, 4)
		for range m.Rows {
			for range m.Cols {
				n, err := input.Read(buffer32)
				if err == io.EOF {
					return err
				} else if err != nil {
					return err
				}
				if n != len(buffer32) {
					return fmt.Errorf("not all bytes read: %d", n)
				}
				value := uint32(0)
				for k := 0; k < 4; k++ {
					value <<= 8
					value |= uint32(buffer32[3-k])
				}
				m.Data = append(m.Data, T(math.Float32frombits(value)))
			}
		}

	}
	return nil
}

func dot[T Float](x, y []T) (z T) {
	switch x := any(x).(type) {
	case []float64:
		switch y := any(y).(type) {
		case []float64:
			for i := range x {
				z += T(x[i] * y[i])
			}
		}
	case []float32:
		switch y := any(y).(type) {
		case []float32:
			z = T(vector.Dot(x, y))
		}
	}
	return z
}

func softmax[T Float](values []T) {
	max := T(0.0)
	for _, v := range values {
		if v > max {
			max = v
		}
	}
	s := max * S
	sum := T(0.0)
	for j, value := range values {
		values[j] = T(math.Exp(float64(value - s)))
		sum += values[j]
	}
	for j, value := range values {
		values[j] = value / sum
	}
}

// CS implements cosine similarity
func (m Matrix[T]) CS(n Matrix[T]) T {
	var sum T
	var count T
	for i := 0; i < len(m.Data); i += m.Cols {
		md := m.Data[i : i+m.Cols]
		nd := n.Data[i : i+m.Cols]
		ab, aa, bb := dot(md, nd), dot(md, md), dot(nd, nd)
		if aa <= 0 || bb <= 0 {
			continue
		}
		sum += ab / (T(math.Sqrt(float64(aa))) * T(math.Sqrt(float64(bb))))
		count++
	}
	return sum / count
}

// SelfAttention computes the self attention of Q, K, V
func SelfAttention[T Float](Q, K, V Matrix[T]) Matrix[T] {
	o := Matrix[T]{
		Size: Size{
			Cols: V.Cols,
			Rows: K.Rows,
		},
		Data: make([]T, 0, V.Rows*K.Rows),
	}
	outputs, values := make([]T, V.Cols), make([]T, Q.Rows)
	V = V.T()
	for i := 0; i < K.Rows; i++ {
		K := K.Data[i*K.Cols : (i+1)*K.Cols]
		for j := 0; j < Q.Rows; j++ {
			Q := Q.Data[j*Q.Cols : (j+1)*Q.Cols]
			values[j] = dot(K, Q)
		}
		softmax(values)

		for j := 0; j < V.Rows; j++ {
			V := V.Data[j*V.Cols : (j+1)*V.Cols]
			outputs[j] = dot(values, V)
		}
		o.Data = append(o.Data, outputs...)
	}
	return o
}

// RNG is a random number generator
type RNG uint32

// LFSRMask is a LFSR mask with a maximum period
const LFSRMask = 0x80000057

// Next returns the next random number
func (r *RNG) Next() uint32 {
	lfsr := *r
	lfsr = (lfsr >> 1) ^ (-(lfsr & 1) & LFSRMask)
	*r = lfsr
	return uint32(lfsr)
}

// Float32 returns a uniform float32
func (r *RNG) Float32() float32 {
	return float32(r.Next()) / float32(math.MaxUint32)
}

// Intn return a uniform random number less than n
func (r *RNG) Intn(n int) int {
	max := uint32((1 << 32) - 1 - (1<<32)%uint64(n))
	v := r.Next()
	for v > max {
		v = r.Next()
	}
	return int(v % uint32(n))
}

func PageRank[T Float](a float32, e int, seed uint32, adj Matrix[T]) Matrix[T] {
	for i := range adj.Rows {
		var sum T
		for ii := range adj.Cols {
			sum += adj.Data[i*adj.Cols+ii]
		}
		for ii := range adj.Cols {
			adj.Data[i*adj.Cols+ii] /= sum
		}
	}
	rng := RNG(seed)
	counts := make([]uint64, adj.Cols)
	iterations := adj.Rows * adj.Cols
	done := make(chan bool, 8)
	process := func(seed uint32) {
		rng, node := RNG(seed), rng.Intn(adj.Cols)
		for range iterations {
			if rng.Float32() > a {
				node = rng.Intn(adj.Cols)
			}
			total, selected, found := T(0.0), T(rng.Float32()), false
			for i, weight := range adj.Data[node*adj.Cols : (node+1)*adj.Cols] {
				total += weight
				if selected < total {
					node, found = i, true
					break
				}
			}
			if !found {
				node = rng.Intn(adj.Cols)
			}
			counter := &counts[node]
			atomic.AddUint64(counter, 1)
		}
		done <- true
	}

	index, flights, cpus := 0, 0, runtime.NumCPU()
	for index < e && flights < cpus {
		go process(rng.Next())
		index++
		flights++
	}
	for index < e {
		<-done
		flights--

		go process(rng.Next())
		index++
		flights++
	}
	for range flights {
		<-done
	}

	p := NewMatrix[T](len(counts), 1)
	for _, value := range counts {
		p.Data = append(p.Data, T(value)/T(e*iterations))
	}
	return p
}

// Transformer implements transform inference
func Transformer[T Float](set Set[T], inputs, outputs Matrix[T]) Matrix[T] {
	itags := set.Named("itags")
	if inputs.Rows != itags.Rows {
		panic("rows should be the same")
	}
	in := NewMatrix[T](itags.Cols+inputs.Cols, inputs.Rows)
	for i := 0; i < itags.Rows; i++ {
		in.Data = append(in.Data, itags.Data[i*itags.Cols:(i+1)*itags.Cols]...)
		in.Data = append(in.Data, inputs.Data[i*inputs.Cols:(i+1)*inputs.Cols]...)
	}
	otags := set.Named("otags")
	if outputs.Rows != otags.Rows {
		panic("rows should be the same")
	}
	out := NewMatrix[T](otags.Cols+outputs.Cols, outputs.Rows)
	for i := 0; i < otags.Rows; i++ {
		out.Data = append(out.Data, otags.Data[i*otags.Cols:(i+1)*otags.Cols]...)
		out.Data = append(out.Data, outputs.Data[i*outputs.Cols:(i+1)*outputs.Cols]...)
	}
	embeddingIn := set.Named("lembeddingIn").MulT(in).Add(set.Named("bembeddingIn")).ReLu()
	formIn := SelfAttention(set.Named("inQ").MulT(embeddingIn),
		set.Named("inK").MulT(embeddingIn),
		set.Named("inV").MulT(embeddingIn)).
		Add(embeddingIn)
	l1In := set.Named("l1In").MulT(formIn).Add(set.Named("b1In")).ReLu().Add(formIn)

	embeddingOut := set.Named("lembeddingOut").MulT(out).Add(set.Named("bembeddingOut")).ReLu()
	formOut := SelfAttention(set.Named("outQ1").MulT(embeddingOut),
		set.Named("outK1").MulT(embeddingOut),
		set.Named("outV1").MulT(embeddingOut)).
		Add(embeddingOut)
	formOut1 := SelfAttention(set.Named("outQ2").MulT(formOut),
		set.Named("outK2").MulT(l1In),
		set.Named("outV2").MulT(l1In)).
		Add(formOut)
	l1Out := set.Named("l1Out").MulT(formOut1).Add(set.Named("b1Out")).ReLu().Add(formOut1)
	return set.Named("linear").MulT(l1Out).Softmax(1)
}
