#pragma once
#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include "hmath.h"
#include "mfcc.h"
#include "sigProcess.h"
#include <time.h>

//#if _WIN32
//#define FFI_PLUGIN_EXPORT __declspec(dllexport)
//#else
//#define FFI_PLUGIN_EXPORT
//#endif

typedef struct {
	Vector reserve_waveforms;
	Vector input_cache;
	Vector lfr_splice_cache;
	Vector waveforms;
	Vector fbanks;
	Vector fbanks_lens;
}Cache;

void WavFrontend(float* s, float** o, int sampleNum);

void WavFrontendOnline(float* s, float** o, int sampleNum, int is_final);

float random_gaussian();

float cal_mean(Vector data, int wlen);

float cal_energy(Vector data, int wlen);


void SubtractColumnMean(float** data, int axis1, int axis2);

void apply_lfr(float** data, float** output, int lfr_m, int lfr_n, int m, int bankNum);

void apply_lfr_online(float** data, float** output, int lfr_m, int lfr_n, int m, int bankNum, int is_final);

float** create2dVector(int axis1, int axis2);

void apply_cmvn(float** data, int axis1, int axis2);

void free2dVector(float** vector, int axis1);