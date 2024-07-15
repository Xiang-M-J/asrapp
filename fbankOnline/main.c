#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include "hmath.h"
#include "mfcc.h"
#include "sigProcess.h"
#include "fbank.h"


#define maxBuffLength 600

/*含义
pi
wlen 窗长度
inc 位移量
bankNum mel滤波器组的个数
MFCCNum MFCC参量的个数
delwin 计算加速系数时的窗口大小
energyFlag 是否计算能量
zeroCrossingFlag 是否计算过零率
brightFlag 是否计算谱中心
subBankEFlag 是否计算子带能量以及个数
regreOrder 加速系数的阶数，1为静态参量，2为静态参量和一阶加速系数，3为静态参量和一阶加速系数和二阶加速系数
*/

/*  最后参量的排列分别为 ( MFCCNum+energyFlag+zeroCrossingFlag+brightFlag+subBankEFlag )*regreOrder  */
/*  最后参量的排列分别为 ( MFCC+能量+过零率+谱中心+子带能量 )*阶数  */

/*程序基本结构
1.读取pcm文件,并转化为十进制
2.计算MFCC系数以及其他参量
3.能量归一化
4.计算加速系数
5.MFCC参量归一归整
6.写入目标文件
*/

typedef struct {
	int sampleRate;
	float samplePeriod;
	float hipassfre;
	float lowpassfre;
	float preemphasise;
	int wlen;
	int inc;
	int fbankFlag;
	int bankNum;
	int MFCCNum;
	int delwin;
	int energyFlag;
	int zeroCrossingFlag;
	int brightFlag;
	int subBandEFlag;
	int regreOrder;
	int MFCC0thFlag;
	int fftLength;
	int saveType;
	int vecNum;
	int channels;
	int sampleNum;
	char fileList[maxBuffLength];
}Config;

static int handler(void* user, const char* section, const char* name, const char* value) {
	Config* pconfig = (Config*)user;
#define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0
	if (MATCH("Frame", "sampleRate")) { pconfig->sampleRate = atoi(value); }
	else if (MATCH("Frame", "hipassfre")) { pconfig->hipassfre = atof(value); }
	else if (MATCH("Frame", "lowpassfre")) { pconfig->lowpassfre = atof(value); }
	else if (MATCH("Frame", "preemphasise")) { pconfig->preemphasise = atof(value); }
	else if (MATCH("Frame", "wlen")) { pconfig->wlen = atoi(value); }
	else if (MATCH("Frame", "inc")) { pconfig->inc = atoi(value); }
	else if (MATCH("Frame", "saveType")) {
		if (strcmp(value, "f\0") == 0)pconfig->saveType = 0;
		else if (strcmp(value, "n\0") == 0) pconfig->saveType = 2;
		else pconfig->saveType = 1;
	}
	else if (MATCH("Frame", "vecNum")) { pconfig->vecNum = atoi(value); }
	else if (MATCH("Frame", "fileList")) { strcpy(pconfig->fileList, value); }
	else if (MATCH("MFCC", "fbankFlag")) { pconfig->fbankFlag = atoi(value); }
	else if (MATCH("MFCC", "bankNum")) { pconfig->bankNum = atoi(value); }
	else if (MATCH("MFCC", "MFCCNum")) { pconfig->MFCCNum = atoi(value); }
	else if (MATCH("MFCC", "MFCC0thFlag")) { pconfig->MFCC0thFlag = atoi(value); }
	else if (MATCH("Others", "energyFlag")) { pconfig->energyFlag = atoi(value); }
	else if (MATCH("Others", "zeroCrossingFlag")) { pconfig->zeroCrossingFlag = atoi(value); }
	else if (MATCH("Others", "brightFlag")) { pconfig->brightFlag = atoi(value); }
	else if (MATCH("Others", "subBandEFlag")) { pconfig->subBandEFlag = atoi(value); }
	else if (MATCH("Others", "fftLength")) { pconfig->fftLength = atoi(value); }
	else if (MATCH("Regression", "regreOrder")) { pconfig->regreOrder = atoi(value); }
	else if (MATCH("Regression", "delwin")) { pconfig->delwin = atoi(value); }
	else return 0;
	return 1;
}

void test_rft() {
	float* waveform;
	int sampleNum = 32;
	waveform = CreateVector(sampleNum);
	for (int i = 0; i < sampleNum; i++)
	{
		*(waveform + i + 1) = (sin(0.01 * i));
	}
	Realft(waveform);
	for (size_t i = 1; i <= sampleNum; i++)
	{
		printf("%f, ", waveform[i]);   // 输出为
	}
	free(waveform);
}

void test_gassuian() {
	float arr[500] = { 0. };
	for (size_t i = 0; i < 500; i++)
	{
		arr[i] = random_gaussian();
	}

	int counter[5] = { 0 };
	for (size_t i = 0; i < 500; i++)
	{
		if (arr[i] < 1 && arr[i] > -1)
		{
			counter[0] += 1;
		}
		else if (arr[i] > 1 && arr[i] < 3)
		{
			counter[1] += 1;
		}
		else if (arr[i] > 3)
		{
			counter[2] += 1;
		}
		else if (arr[i] < -1 && arr[i] > -3)
		{
			counter[3] += 1;
		}
		else {
			counter[4] += 1;
		}
	}
	printf("%d, %d, %d, %d, %d", counter[4], counter[3], counter[0], counter[1], counter[2]);
}

void test_wave2fbank() {
	float* waveform;
	float** output = NULL;
	int bankNum = 80;
	int sampleNum = 208832;
	int lfr_m = 5, lfr_n = 1;
	int m = 1 + (sampleNum - 400) / 160;
	int m_lfm = (int)((lfr_m - 1) / 2);
	int axis1 = (int)ceil(m + m_lfm - (int)((lfr_m - 1) / 2) / (lfr_n * 1.0));
	waveform = (float*)malloc(sizeof(float) * (sampleNum));
	output = create2dVector(axis1, bankNum * lfr_m);
	for (int i = 0; i < sampleNum; i++)
	{
		waveform[i] = 100 * (sin(0.01 * i));
	}
	
	WavFrontendOnline(waveform, output, sampleNum);

	for (size_t i = 0; i < axis1; i++)
	{
		for (size_t j = 0; j < bankNum * lfr_m; j++)
		{
			printf("%f,", output[i][j]);
		}
		printf("\n\n\n");
	}
	

	free(waveform);
	free2dVector(output, 3);
	return;
}

int main(int argc, char** argv) {
	//test_rft();
	test_wave2fbank();
	//test_gassuian();
}
