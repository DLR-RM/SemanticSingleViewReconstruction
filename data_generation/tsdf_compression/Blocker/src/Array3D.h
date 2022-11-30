//
// Created by Maximilian Denninger on 2019-06-17.
//

#ifndef LOSSCALCULATOR_ARRAY3D_H
#define LOSSCALCULATOR_ARRAY3D_H

#include <vector>
#include <iostream>
#include <list>
#include <cmath>
#include <array>
#include "StopWatch.h"


template<typename T>
class Array3D {
public:
	Array3D(){
		m_innerSize = 0;
		m_size = 0;
		m_values.resize(1);
	}

	explicit Array3D(unsigned int size){
		m_size = size;
		m_innerSize = size * size * size;
		m_values.resize(m_innerSize);
	}

	template <typename ArrayType>
	Array3D(ArrayType* array, unsigned int size){
		init<ArrayType>(array, size);
	}

	template <typename ArrayType>
	Array3D(Array3D<ArrayType>& array){
		init<ArrayType>(&array.getInner(0), array.length());
	}

	template <typename ArrayType>
	void init(ArrayType* array, unsigned int size){
		m_values.clear();
		m_size = size;
		m_innerSize = size * size * size;
		m_values.resize(m_innerSize);
		for(unsigned int i = 0; i < m_innerSize; ++i){
			m_values[i] = T(array[i]);
		}
	}

	void fill(const T& value){
		for(unsigned int i = 0; i < m_innerSize; ++i){
			m_values[i] = value;
		}
	}

	T& operator()(unsigned int i, unsigned int j, unsigned int k){
		return m_values[i * m_size * m_size + j * m_size + k];
	}

	const T& operator()(unsigned int i, unsigned int j, unsigned int k) const {
		return m_values[i * m_size * m_size + j * m_size + k];
	}

	unsigned int length() const {
		return m_size;
	}
	unsigned int innerSize(){
		return m_innerSize;
	}

	bool isInCube(const int first, const int second, const int third){
	    /**
	     * Checks if a given pose can be used as an index to the array
	     */
		return first >= 0 && first < m_size && second >= 0 &&
		       second < m_size && third >= 0 && third < m_size;
	}

	T& getInner(unsigned int i){
		return m_values[i];
	}

private:

	std::vector<T> m_values;
	unsigned int m_innerSize;
	unsigned int m_size;

};




#endif //LOSSCALCULATOR_ARRAY3D_H
