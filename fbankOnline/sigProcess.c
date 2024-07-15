#include "sigProcess.h"

/*circle shift the signal*/
void circshift(Vector v, int shift)
{
	int i = 1; Vector v_temp = CreateVector(VectorSize(v));
	if (shift < 0)do { shift += VectorSize(v); } while (shift < 0);
	if (shift >= VectorSize(v))do { shift -= VectorSize(v); } while (shift >= VectorSize(v));
	for (i = 1; (i + shift) <= VectorSize(v); i++)v_temp[i + shift] = v[i];
	for (; i <= VectorSize(v); i++)v_temp[i + shift - VectorSize(v)] = v[i];
	CopyVector(v_temp, v);
	FreeVector(v_temp);
}

/*find the first index of abs|sample| exceeding the thre from the front or end*/
int find(Vector v, float thre, int FrontOrEndFlag)
{
	int i; int m = 0;
	if (FrontOrEndFlag == 1) {
		for (i = 1; i <= VectorSize(v); i++)if (fabs(v[i]) > thre) { m = i; break; }
		return m;
	}
	else if (FrontOrEndFlag == -1) {
		for (i = VectorSize(v); i >= 1; i--)if (fabs(v[i]) > thre) { m = i; break; }
		return m;
	}
	else
		return 0;
}

void pad_signal(Vector * yP, Vector x, int Npad)
{
	int i = 0; int j = 0;
	int orig_sz = VectorSize (x);
	int Norig = VectorSize(x);
	int end = Norig + (int)floor((float)(Npad - Norig) / 2.0);
	int end2 = (int)ceil((float)(Npad - Norig) / 2.0);
	int end3 = Norig + (int)floor((float)(Npad - Norig) / 2.0) + 1;
	IntVec ind0 = CreateIntVec(2 * Norig);
	IntVec conjugate0 = CreateIntVec(2 * Norig);
	IntVec conjugate = CreateIntVec(Npad);
	IntVec ind = CreateIntVec(Npad);
	IntVec src = CreateIntVec(end - Norig);
	IntVec dst = CreateIntVec(end - Norig);
	ZeroIntVec(ind); ZeroIntVec(conjugate);
	for (i = 1; i <= Norig; i++)ind0[i] = i;
	for (i=Norig; i >= 1; i--)ind0[2 * Norig - i + 1] = i;
	for (i = 1; i <= Norig; i++)conjugate0[i] = 0;
	for (i = Norig + 1; i <= 2 * Norig; i++)conjugate0[i] = 1;
	for (i = 1; i <= Norig; i++)ind[i] = i;
	for (i = 1; i <= VectorSize(src); i++)src[i] = (Norig + i-1) % (VectorSize(ind0)) + 1;
	for (i = 1; i <= VectorSize(dst); i++)dst[i] = Norig + i;
	for (i = 1; i <= VectorSize(src); i++)ind[dst[i]] = ind0[src[i]];
	for (i = 1; i <= VectorSize(src); i++)conjugate[dst[i]] = conjugate0[src[i]];
	FreeIntVec(src); FreeIntVec(dst);
	src = CreateIntVec(end2); dst = CreateIntVec(Npad - end3 + 1);
	for (i = 1; i <= VectorSize(src); i++) {
		if((VectorSize(ind0) - i)>=0)src[i] = ((VectorSize(ind0) - i) % (VectorSize(ind0))) + 1;
		else src[i] = ((VectorSize(ind0) - i+ VectorSize(ind0)) % (VectorSize(ind0))) + 1;
	}
	for (i = Npad, j = 1; i >= end3; i--, j++)dst[j] = i;
	for (i = 1; i <= VectorSize(src); i++)ind[dst[i]] = ind0[src[i]];
	for (i = 1; i <= VectorSize(src); i++)conjugate[dst[i]] = conjugate0[src[i]];
	*yP = CreateVector(VectorSize(ind));
	for (i = 1; i <= VectorSize(ind); i++)(*yP)[i] = x[ind[i]];
	FreeIntVec(ind0); FreeIntVec(conjugate0); FreeIntVec(conjugate); FreeIntVec(ind); FreeIntVec(src); FreeIntVec(dst);
}

void unpad_signal(Vector * yP, Vector x, int res, int target_sz)
{
	int i = 0;
	int padded_sz = VectorSize(x);
	float offset = 0;
	int offset_ds = 0;
	int target_sz_ds = 1 + floor((float)(target_sz - 1) / pow(2.0, (float)res));
	*yP = CreateVector(target_sz_ds);
	for (i = 1; i <= VectorSize(*yP); i++)(*yP)[i] = x[i];
}

Matrix frameRawSignal(IntVec v, int wlen, int inc, float preEmphasiseCoefft, int enableHamWindow)
{
	int numSamples = VectorSize(v);
	int numFrames = (numSamples - (wlen - inc)) / inc;
	Matrix m = NULL; 
	Vector v1 = NULL; Vector HamWindow = NULL;
	int i = 0, j = 0, pos = 1; float a = 0;

	v1 = CreateVector(numSamples);
	for (i = 1; i <= numSamples; i++)v1[i] = (float)v[i];
	PreEmphasise(v1, preEmphasiseCoefft);

	HamWindow = CreateVector(wlen);
	a = 2 * pi / (wlen - 1);
	if(enableHamWindow)for (i = 1; i <= wlen; i++)HamWindow[i] = 0.54 - 0.46 * cos(a*(i - 1));
	else for (i = 1; i <= wlen; i++)HamWindow[i] = 1.0;

	if ((numSamples - (inc - wlen)) % inc != 0)numFrames++;
	m = CreateMatrix(numFrames, wlen);
	for (i = 1; i <= numFrames; i++) {
		pos = (i - 1)*inc + 1;
		for (j = 1; j <= wlen; j++,pos++) {
			if (pos > numSamples)m[i][j] = 0.0;
			else m[i][j] = (float)v1[pos]*HamWindow[j];
		}
	}
	FreeVector(v1); FreeVector(HamWindow);
	return m;
}

/* EXPORT->PreEmphasise: pre-emphasise signal in s */
void PreEmphasise(Vector s, float k)
{
	int i;
	float preE;

	preE = k;
	if (k == 0.0)return;
	for (i = VectorSize(s); i >= 2; i--)
		s[i] -= s[i - 1] * preE;
	s[1] *= 1.0 - preE;
}

float calBrightness(Vector fftx)
{
	int i;
	float sum = 0.0;
	float te = 0.0;
	float b = 0;
	if (((int)VectorSize(fftx)) % 2 != 0)printf("something wrong in cal brightness");
	for (i = 1; i <= ((int)VectorSize(fftx)) / 2; i++) {
		sum += (fftx[2 * i - 1] * fftx[2 * i - 1] + fftx[2 * i] * fftx[2 * i])*(float)i;
		te += fftx[2 * i - 1] * fftx[2 * i - 1] + fftx[2 * i] * fftx[2 * i];
	}
	b = sum / te;
	b = b / ((float)VectorSize(fftx) / 2.0);
	return b;
}


void calSubBankE(Vector fftx, Vector subBankEnergy)
{
	int i;
	int numBank = VectorSize(subBankEnergy); int bankSize = (int)VectorSize(fftx) / (2 * numBank);
	int bankNum = 1;
	float te = 0.0;
	float sum = 0.0;
	for (i = 1; i <= (int)VectorSize(fftx) / 2; i++)te+= fftx[2 * i - 1] * fftx[2 * i - 1] + fftx[2 * i] * fftx[2 * i];
	for (i = 1; i <= (int)VectorSize(fftx) / 2; i++) {
		if (i <= bankNum*bankSize) {
			sum += fftx[2 * i - 1] * fftx[2 * i - 1] + fftx[2 * i] * fftx[2 * i];
		}
		else {
			subBankEnergy[bankNum] = sum / te;
			//printf("sum: %f\n", sum/te);
			bankNum++; sum = 0.0; i--;
		}
	}
	subBankEnergy[bankNum] = sum / te;

}

/*Z-normalization, Not tested */
void ZNormalize(float * data, int vSize, int n, int step)
{
	float sum1,sum2;
	float *fp, sd,mean;
	int i,j;
	for (i = 0; i < vSize; i++) {
		sum1 = 0.0;sum2=0.0;
		fp = data + i;
		for (j = 0; j < n; j++) { sum1+=(*fp);sum2 += (*fp)*(*fp); fp += step; }
		mean=sum1/(float)n;
		sd = sqrt(sum2 / (float)n-mean*mean);
		fp = data + i;
		for (j = 0; j < n; j++) {
			*fp = ((*fp)-mean)/sd; fp += step;
		}
	}
}


/* GenHamWindow: generate precomputed Hamming window function */
Vector GenHamWindow(int frameSize)
{
	int i;
	float a;
	Vector hamWin = CreateVector(frameSize);

	a = 2 * pi / (frameSize - 1);
	for (i = 1; i <= frameSize; i++)
		hamWin[i] = 0.54 - 0.46 * cos(a*(i - 1));
	return hamWin;
}

/* EXPORT->Ham: Apply Hamming Window to Speech frame s */
void Ham(Vector s, Vector hamWin,int hamWinSize)
{
	int i, frameSize;
	frameSize = VectorSize(s);
	if (hamWinSize != frameSize)
		GenHamWindow(frameSize);
	for (i = 1; i <= frameSize; i++) {
		s[i] *= hamWin[i];
		//		printf("%d %f\n", i,s[i]);
	}
}
