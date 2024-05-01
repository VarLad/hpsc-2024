#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <x86intrin.h>

int main() {
  const int N = 20;
  float x[N], y[N], m[N], fx[N], fy[N], ran[N];

  for (int i = 0; i < N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    ran[i] = i;
  }

  __m512 ran_list = _mm512_load_ps(ran);
  __m512 z = _mm512_setzero_ps();
  
  for (int i = 0; i < N; i++) {
    __m512 ivec = _mm512_sub_ps(_mm512_set1_ps(i), ran_list);
    __m512 xvec = _mm512_sub_ps(_mm512_set1_ps(x[i]), _mm512_load_ps(x));
    __m512 yvec = _mm512_sub_ps(_mm512_set1_ps(y[i]), _mm512_load_ps(y));

    // mask
    __mmask16 mask1 = _mm512_cmp_ps_mask(ivec, z, _MM_CMPINT_NE);
    __m512 rvecx = _mm512_mask_blend_ps(mask1, z, xvec);
    __m512 rvecy = _mm512_mask_blend_ps(mask1, z, yvec);
    __m512 rvec = _mm512_add_ps(_mm512_mul_ps(rvecx, rvecx), _mm512_mul_ps(rvecy, rvecy));
    __mmask16 mask2 = _mm512_cmp_ps_mask(rvec, z, _MM_CMPINT_EQ);
    rvec = _mm512_rsqrt14_ps(_mm512_mask_blend_ps(mask2, rvec, _mm512_set1_ps(1)));
    rvec = _mm512_mul_ps(_mm512_mul_ps(rvec, rvec), rvec);
    rvec = _mm512_mask_blend_ps(mask2, rvec, z);

    __m512 mvec = _mm512_load_ps(m);
    fx[i] = _mm512_reduce_add_ps(_mm512_mask_sub_ps(z, mask1, z, _mm512_mul_ps(_mm512_mul_ps(rvecx, rvec), mvec)));
    fy[i] = _mm512_reduce_add_ps(_mm512_mask_sub_ps(z, mask1, z, _mm512_mul_ps(_mm512_mul_ps(rvecy, rvec), mvec)));

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}
