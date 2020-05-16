%% Examples of benchmarks for different input formats
addpath benchmarks
clear all;close all;clc;


%% region benchmarks for results stored as a cell of segmentations

imgDir = '../data/images';
gtDir = '../data/groundTruth';
inDir = '../data/segs';
outDir = '../data/test_5';
mkdir(outDir);
nthresh = 5;

tic;
regionBench(imgDir, gtDir, inDir, outDir, nthresh);
toc;

