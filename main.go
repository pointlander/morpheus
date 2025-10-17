// Copyright 2025 The Morpheus Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"flag"
	"fmt"
	"github.com/alixaxel/pagerank"
	"io"
	"math"
	"math/rand"
	"os"
	"regexp"
	"runtime/pprof"
	"sort"
	"strings"

	"github.com/pointlander/morpheus/kmeans"
)

var (
	// FlagIris is the iris clusterting mode
	FlagIris = flag.Bool("iris", false, "iris clustering")
	// FlagIrisMarkov is the iris markov model
	FlagIrisMarkov = flag.Bool("irismarkov", false, "iris markov mode")
	// FlagClass classifies text
	FlagClass = flag.Bool("class", false, "classify text")
	// FlagText text generation mode
	FlagText = flag.Bool("text", false, "text generation mode")
	// FlagE is an experiment
	FlagE = flag.Bool("e", false, "experiment")
	// FlagTxt txt mode
	FlagTxt = flag.Bool("txt", false, "txt mode")
	// Flag3m markov mcts morpheus mode
	Flag3m = flag.Bool("3m", false, "markov mcts morpheus mode")
	// FlagMach1 mach1 mode
	FlagMach1 = flag.Bool("mach1", false, "mach 1 mode")
	// FlagExtreme extreme mode
	FlagExtreme = flag.Bool("extreme", false, "extreme mode")
	// FlagPageRank pagerank mode
	FlagPageRank = flag.Bool("pagerank", false, "pagerank mode")
	// FlagRandom is the random matrix mode
	FlagRandom = flag.Bool("random", false, "flag random")
	// FlagLearn learn the vector database
	FlagLearn = flag.Bool("learn", false, "learn the vector database")
	// FlagPrompt the prompt to use
	FlagPrompt = flag.String("prompt", "What is the meaning of life?", "the prompt to use")
	// cpuprofile profiles the program
	cpuprofile = flag.String("cpuprofile", "", "write cpu profile to `file`")
)

const (
	order = 4
)

type Markov [order]byte
type Model [order]map[Markov][]uint32

// Lookup looks a vector up
func Lookup(markov *[order]Markov, model *Model) []float32 {
	for i := range markov {
		i = order - 1 - i
		vector := model[i][markov[i]]
		if vector != nil {
			sum := float32(0.0)
			for _, value := range vector {
				sum += float32(value)
			}
			result := make([]float32, len(vector))
			for ii, value := range vector {
				result[ii] = float32(value) / sum
			}
			return result
		}
	}
	return nil
}

// Iterate iterates a markov model
func Iterate(markov *[order]Markov, state byte) {
	for i := range markov {
		state := state
		for ii, value := range markov[i][:i+1] {
			markov[i][ii], state = state, value
		}
	}
}

// PageRank mode
func PageRankMode() {
	type File struct {
		Name string
		Data []byte
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
	load := func(book string) []byte {
		file, err := Text.Open(book)
		if err != nil {
			panic(err)
		}
		defer file.Close()
		breader := bzip2.NewReader(file)
		data, err := io.ReadAll(breader)
		if err != nil {
			panic(err)
		}
		return data
	}
	for i := range files {
		files[i].Data = load(fmt.Sprintf("books/%s", files[i].Name))
	}

	type Word struct {
		Word  string
		Count int
	}

	bible := string(files[1].Data)
	reg := regexp.MustCompile(`\s+`)
	parts := reg.Split(bible, -1)
	reg = regexp.MustCompile(`[\p{P}]+|[\d]`)
	unique := make(map[string]*Word)
	links := make(map[string]map[string]uint64)
	previous, current := "", ""
	for _, part := range parts {
		parts := reg.Split(part, -1)
		parts = append(parts, reg.FindAllString(part, -1)...)
		for _, part := range parts {
			word := strings.ToLower(strings.TrimSpace(part))
			count := unique[word]
			if count == nil {
				count = &Word{
					Word: word,
				}
			}
			count.Count++
			unique[word] = count

			previous, current = current, word
			link := links[previous]
			if link == nil {
				link = make(map[string]uint64)
			}
			link[current]++
			links[previous] = link
		}
	}
	words := make([]*Word, len(unique))
	{
		count := 0
		for _, word := range unique {
			words[count] = word
			count++
		}
	}
	sort.Slice(words, func(i, j int) bool {
		if words[i].Count == words[j].Count {
			return words[i].Word < words[j].Word
		}
		return words[i].Count > words[j].Count
	})
	fmt.Println(len(words))
	length := 2 * 1024
	words = words[:length]

	type Path struct {
		Path []int
		Cost float64
	}
	done := make(chan Path)
	process := func(seed int64) {
		context, cost := []int{}, 0.0
		for _, word := range []string{"the", "lord", "is", "good"} {
			for i := range words {
				if words[i].Word == word {
					context = append(context, i)
					break
				}
			}
		}

		rng := rand.New(rand.NewSource(seed))
		for range 33 {
			size := len(words) + len(context)
			graph := pagerank.NewGraph()
			adjacency := NewMatrix(size, size, make([]float32, size*size)...)
			for i := range words {
				from := links[words[i].Word]
				for ii := range words {
					to := from[words[ii].Word]
					graph.Link(uint32(i), uint32(ii), float64(to))
					adjacency.Data[i*adjacency.Cols+ii] = float32(to)
				}
			}
			for i, word := range context {
				for ii := range words {
					sum := float32(0.0)
					for _, value := range adjacency.Data[ii*adjacency.Cols : ii*adjacency.Cols+length] {
						sum += value
					}
					graph.Link(uint32(ii), uint32(length+i), float64(sum/8))
					adjacency.Data[ii*adjacency.Cols+length+i] += sum / 8
				}
				copy(adjacency.Data[(length+i)*adjacency.Cols:(length+i+1)*adjacency.Cols],
					adjacency.Data[word*adjacency.Cols:(word+1)*adjacency.Cols])
				if i > 0 {
					sum := float32(0.0)
					for _, value := range adjacency.Data[(length+i-1)*adjacency.Cols : (length+i-1)*adjacency.Cols+length] {
						sum += value
					}
					graph.Link(uint32(length+i-1), uint32(length+i), float64(sum/8))
					adjacency.Data[(length+i-1)*adjacency.Cols+length+i] += sum / 8
				}
			}
			//result := PageRank(1.0, 33, rng.Uint32(), adjacency)
			result := make([]float64, size)
			graph.Rank(1.0, 1e-3, func(node uint32, rank float64) {
				result[node] = rank
			})
			distribution, sum := make([]float64, len(words)), 0.0
			for _, value := range result[:length] {
				if value < 0 {
					value = -value
				}
				sum += float64(value)
			}
			for iii, value := range result[:length] {
				if value < 0 {
					value = -value
				}
				distribution[iii] = float64(value) / sum
			}

			total, selected := 0.0, rng.Float64()
			for iii, value := range distribution {
				total += value
				if selected < total {
					cost += value
					context = append(context, iii)
					break
				}
			}
		}
		done <- Path{
			Path: context,
			Cost: cost,
		}
	}

	rng := rand.New(rand.NewSource(1))
	for range 8 {
		go process(rng.Int63())
	}
	pathes := make([]Path, 8)
	for i := range 8 {
		pathes[i] = <-done
	}
	sort.Slice(pathes, func(i, j int) bool {
		return pathes[i].Cost < pathes[j].Cost
	})
	for _, path := range pathes {
		context := path.Path
		for _, word := range context {
			fmt.Printf("%s ", words[word].Word)
		}
		fmt.Println()
	}
}

// RandomMode random mode
func RandomMode() {
	rng := rand.New(rand.NewSource(1))
	iris := Load()
	rl1 := NewMatrix(4, 4, make([]float64, 4*4)...)
	for i := range rl1.Data {
		rl1.Data[i] = rng.NormFloat64()
	}
	l1 := rl1.GramSchmidt().T()
	const size = 8
	rl2 := NewMatrix(size, size, make([]float64, size*size)...)
	for i := range rl2.Data {
		rl2.Data[i] = rng.NormFloat64()
	}
	l2 := rl2.GramSchmidt().T()
	data := make([][]float64, len(iris))
	type Row struct {
		Fisher
		Embedding []float64
	}
	rows := make([]Row, len(iris))
	for i, row := range iris {
		input := NewMatrix(4, 1, row.Measures...)
		x := l1.MulT(input).Everett()
		y := l2.MulT(x)
		fmt.Println(row.Label, y.Data)
		data[i] = y.Data
		rows[i].Fisher = row
		rows[i].Embedding = y.Data
	}
	avg := make([]float64, size)
	for _, row := range data {
		for key, value := range row {
			avg[key] += value
		}
	}
	for key, value := range avg {
		avg[key] = value / float64(len(iris))
	}
	stddev := make([]float64, size)
	for _, row := range data {
		for key, value := range row {
			diff := avg[key] - value
			stddev[key] += diff * diff
		}
	}
	for key, value := range stddev {
		stddev[key] = math.Sqrt(value / float64(len(iris)))
	}
	type Column struct {
		Index  int
		Stddev float64
	}
	columns := make([]Column, len(stddev))
	for key, value := range stddev {
		columns[key].Index = key
		columns[key].Stddev = value
	}
	sort.Slice(columns, func(i, j int) bool {
		return columns[i].Stddev > columns[j].Stddev
	})
	compressed := make([][]float64, len(data))
	for i := range compressed {
		for _, value := range columns[:2] {
			compressed[i] = append(compressed[i], data[i][value.Index])
		}
	}
	meta := make([][]float64, len(iris))
	for i := range meta {
		meta[i] = make([]float64, len(iris))
	}
	const k = 3
	for i := 0; i < 33; i++ {
		clusters, _, err := kmeans.Kmeans(int64(i+1), compressed, k, kmeans.SquaredEuclideanDistance, -1)
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
	clusters, _, err := kmeans.Kmeans(1, meta, k, kmeans.SquaredEuclideanDistance, -1)
	if err != nil {
		panic(err)
	}
	for i, row := range rows {
		fmt.Println(row.Label, clusters[i])
		rows[i].Cluster = clusters[i]
	}
	for i := range columns {
		fmt.Println(columns[i])
	}
	a := make(map[string][3]int)
	for i := range rows {
		histogram := a[rows[i].Label]
		histogram[rows[i].Cluster]++
		a[rows[i].Label] = histogram
	}
	for k, v := range a {
		fmt.Println(k, v)
	}

	type Pivot struct {
		Max   float64
		Col   int
		Row   int
		Left  *Pivot
		Right *Pivot
	}
	cols := make(map[int]bool, size)
	var process func(depth int, rows []Row) *Pivot
	process = func(depth int, rows []Row) *Pivot {
		pivot := Pivot{}
		for ii := range size {
			if cols[ii] {
				continue
			}
			sort.Slice(rows, func(i, j int) bool {
				return rows[i].Embedding[ii] < rows[j].Embedding[ii]
			})
			for iii := 1; iii < len(rows); iii++ {
				avgA, cA := 0.0, 0.0
				for iv := 0; iv < iii; iv++ {
					avgA += rows[iv].Embedding[ii]
					cA++
				}
				avgA /= cA
				varA := 0.0
				for iv := 0; iv < iii; iv++ {
					diff := avgA - rows[iv].Embedding[ii]
					varA += diff * diff
				}
				varA /= cA

				avgB, cB := 0.0, 0.0
				for iv := iii; iv < len(rows); iv++ {
					avgB += rows[iv].Embedding[ii]
					cB++
				}
				avgB /= cB
				varB := 0.0
				for iv := iii; iv < len(rows); iv++ {
					diff := avgB - rows[iv].Embedding[ii]
					varB += diff * diff
				}
				varB /= cB

				avg, c := 0.0, 0.0
				for iv := 0; iv < len(rows); iv++ {
					avg += rows[iv].Embedding[ii]
					c++
				}
				avg /= c
				v := 0.0
				for iv := 0; iv < len(rows); iv++ {
					diff := avg - rows[iv].Embedding[ii]
					v += diff * diff
				}
				v /= c

				if gain := v - (varA + varB); gain > pivot.Max {
					pivot.Max, pivot.Col, pivot.Row = gain, ii, iii
				}
			}
		}
		cols[pivot.Col] = true
		if depth == 0 {
			return &pivot
		}
		sort.Slice(rows, func(i, j int) bool {
			return rows[i].Embedding[pivot.Col] < rows[j].Embedding[pivot.Col]
		})
		pivot.Left = process(depth-1, rows[:pivot.Row])
		pivot.Right = process(depth-1, rows[pivot.Row:])
		return &pivot
	}
	pivots := process(1, rows)
	x := pivots
	sort.Slice(rows, func(i, j int) bool {
		return rows[i].Embedding[x.Col] < rows[j].Embedding[x.Col]
	})
	s1 := rows[:x.Row]
	r := rows[x.Row:]
	y := pivots.Right
	if pivots.Left.Max > y.Max {
		y = pivots.Left
		s1 = rows[x.Row:]
		r = rows[:x.Row]
	}
	sort.Slice(r, func(i, j int) bool {
		return r[i].Embedding[x.Col] < r[j].Embedding[x.Col]
	})
	s2 := r[:y.Row]
	s3 := r[y.Row:]
	for i := range s1 {
		s1[i].Cluster = 0
	}
	for i := range s2 {
		s2[i].Cluster = 1
	}
	for i := range s3 {
		s3[i].Cluster = 2
	}
	for _, row := range rows {
		fmt.Println(row.Cluster, row.Label)
	}
	a = make(map[string][3]int)
	for i := range rows {
		histogram := a[rows[i].Label]
		histogram[rows[i].Cluster]++
		a[rows[i].Label] = histogram
	}
	for k, v := range a {
		fmt.Println(k, v)
	}
}

func main() {
	flag.Parse()

	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			panic(err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			panic(err)
		}
		defer pprof.StopCPUProfile()
	}

	if *FlagIris {
		IrisMode()
		return
	}

	if *FlagIrisMarkov {
		IrisMarkovMode()
		return
	}

	if *FlagClass {
		ClassMode()
		return
	}

	if *FlagText {
		TextMode()
		return
	}

	if *FlagTxt {
		TxtMode()
		return
	}

	if *Flag3m {
		_3mMode()
		return
	}

	if *FlagExtreme {
		ExtremeMode()
		return
	}

	if *FlagMach1 {
		Mach1Mode()
		return
	}

	if *FlagPageRank {
		PageRankMode()
		return
	}

	if *FlagRandom {
		RandomMode()
		return
	}

	rng := rand.New(rand.NewSource(1))

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
			book.Model[i] = make(map[Markov][]uint32)
		}
		for _, value := range data {
			for ii := range markov {
				vector := book.Model[ii][markov[ii]]
				if vector == nil {
					vector = make([]uint32, size)
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

	type String struct {
		String []byte
		Markov [order]Markov
		Reward float64
		Vector [][]float32
	}
	strings := make([]String, 1024)
	for i := range strings {
		strings[i].String = []byte("What is the meaning of life?")
		for _, value := range strings[i].String {
			Iterate(&strings[i].Markov, value)
		}
		for range 128 {
			distribution := Lookup(&strings[i].Markov, &files[1].Model)
			strings[i].Vector = append(strings[i].Vector, distribution)
			sum, selected := float32(0.0), rng.Float32()
			for key, value := range distribution {
				sum += value
				if selected < sum {
					strings[i].String = append(strings[i].String, byte(key))
					Iterate(&strings[i].Markov, byte(key))
					strings[i].Reward += float64(value)
					break
				}
			}
		}
	}
	sort.Slice(strings, func(i, j int) bool {
		return strings[i].Reward > strings[j].Reward
	})
	fmt.Println(string(strings[0].String))
}
