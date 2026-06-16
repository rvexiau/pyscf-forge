#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <omp.h>
//#include "config.h"
//#include "vhf/fblas.h"
//#include "fci.h"

void FCICSFddstrs2csdstrs (uint64_t * csdstrs, uint64_t * ddstrs, size_t nstr, int norb, int neleca, int nelecb)
{

    size_t i;
    int iorb, isorb, ispin;
    uint64_t * astrs = ddstrs;
    uint64_t * bstrs = & ddstrs[nstr];
    uint64_t * npairs = csdstrs;
    uint64_t * dconf_strs = & csdstrs[nstr];
    uint64_t * sconf_strs = & csdstrs[2*nstr];
    uint64_t * spins_strs = & csdstrs[3*nstr];

    for (i = 0; i < nstr; i++){
        npairs[i] = 0;
        dconf_strs[i] = 0;
        sconf_strs[i] = 0;
        spins_strs[i] = 0;

        isorb = 0;
        ispin = 0;

        for (iorb = 0; iorb < norb; iorb++){
            if ((1ULL << iorb) & astrs[i] & bstrs[i]) { /* DOUBLY OCCUPIED */
                npairs[i]++;
                dconf_strs[i] |= 1ULL << iorb; /* This is adding 2 electrons at position iorb */
            } else if ((1ULL << iorb) & (astrs[i] | bstrs[i])) { /* SINGLY OCCUPIED */
                sconf_strs[i] |= 1ULL << isorb; /* This is adding 1 electron at non-double position isorb */
                isorb++;
                if ((1ULL << iorb) & astrs[i]) { spins_strs[i] |= 1ULL << ispin ; } /* This is adding 1 alpha spin state at spin ispin */
                ispin++;
            } else { isorb++; } /* VIRTUAL */
        }
    }
}

void FCICSFcsdstrs2ddstrs (uint64_t * ddstrs, uint64_t * csdstrs, size_t nstr, int norb, int neleca, int nelecb)
{

    size_t i;
    int iorb, isorb, ispin;
    uint64_t * astrs = ddstrs;
    uint64_t * bstrs = & ddstrs[nstr];
    uint64_t * dconf_strs = & csdstrs[nstr];
    uint64_t * sconf_strs = & csdstrs[2*nstr];
    uint64_t * spins_strs = & csdstrs[3*nstr];

    for (i = 0; i < nstr; i++){
        astrs[i] = 0;
        bstrs[i] = 0;

        isorb = 0;
        ispin = 0;

        for (iorb = 0; iorb < norb; iorb++){
            if ((1ULL << iorb) & dconf_strs[i]){
                astrs[i] |= 1ULL << iorb;
                bstrs[i] |= 1ULL << iorb;
            } else {
                if ((1ULL << isorb) & sconf_strs[i]){
                    if ((1ULL << ispin) & spins_strs[i]){
                        astrs[i] |= 1ULL << iorb;
                    } else {
                        bstrs[i] |= 1ULL << iorb;
                    }
                    ispin++;
                }
                isorb++;
            }
        }
    }
}


void FCICSFmakecsf (double * umat, uint64_t * detstr, uint64_t * coupstr, int nspin, size_t ndet, size_t ncoup, int twoS, int twoMS)
{


#pragma omp parallel default(shared)
{

    size_t idet, icoup, ispin, idetcoup, ndetcoup;
    int track2S, track2MS, sgn, osgn;
    uint64_t sup, msup;
    double numerator, denominator;

    ndetcoup = ndet * ncoup;

#pragma omp for schedule(static) 

    for (idetcoup = 0; idetcoup < ndetcoup; idetcoup++){
        /* This is a shitty way to do a nested loop but the gcc version that pyscf has to stay compatible with can't use "collapse" statements */
        idet = idetcoup / ncoup;
        icoup = idetcoup % ncoup;
        track2S = 1;
        track2MS = 1ULL & detstr[idet] ? 1 : -1;
        numerator = 1;
        denominator = 1;
        sgn = 1;
        // Commute each spin-down electron past each spin-up electron
        osgn = -track2MS;
        for (ispin = 1; ispin < nspin; ispin++){
            sup = (1ULL << ispin) & coupstr[icoup];
            msup = (1ULL << ispin) & detstr[idet];
            if (msup){ track2MS++; osgn *= -1; } else { track2MS--; sgn *= osgn; }
            /* Clebsch-Gordan coefficient <j1,j2,m1,m2|J,M=m1+m2> (j2 = 1/2, m2 = +-1/2)
 *              = sgn * sqrt (num / denom)
 *              sgn = sgn(J2-j1)^delta(m2,+1/2)
 *              num = j1 + 1/2 + sgn(J-j1)*sgn(m2)*M
 *              denom = 2*j1 + 1 
 *              All numbers are half-integers so multiply num and denom by 2
 *          */
            numerator *= (sup == msup) ? track2S + track2MS + 1 : track2S + 1 - track2MS;
            if (numerator == 0){ break; }
            denominator *= (track2S + 1);
            if (msup && !sup){ sgn *= -1; }

            /* All numbers are half-integers so num is *2 computed in this way.
 *          */
            numerator /= 2;

            if (sup){ track2S++; } else { track2S--; }
        } 
        if (numerator == 0) { umat[idetcoup] = 0.0; } else {
            umat[idetcoup] = sgn * sqrt ((double) numerator / denominator);
        }
    }

}
}


void FCICSFmakeS2mat (double * S2mat, uint64_t * detstr, size_t ndet, int nspin, int twoMS)
{

    size_t idet, jdet;
    int nflip, iflip, osgn, sgn;
    uint64_t flipdet;
    double sz2 = (double) twoMS * twoMS / 4;
    double diag = sz2 + (double) nspin / 2;

    for (idet = 0; idet < ndet; idet++){ for (jdet = 0; jdet < ndet; jdet++){
        flipdet = detstr[idet] ^ detstr[jdet];
        nflip = 0;
        if (flipdet == 0ULL){
            S2mat[idet*ndet + jdet] = diag;
            continue;
        }
        osgn = -1;
        sgn = 1;
        for (iflip = 0; iflip < nspin; iflip++){
            osgn *= -1;
            if ((1ULL << iflip) & detstr[idet]){ sgn *= osgn; }
            if ((1ULL << iflip) & detstr[jdet]){ sgn *= osgn; }
            if ((1ULL << iflip) & flipdet){ nflip++; }
            if (nflip > 2){ break; }
        }
        if (nflip == 2){
            S2mat[idet*ndet + jdet] = sgn * 1.0;
        }
    }}

}


void FCICSFgetscstrs (uint64_t * scstrs, bool * mask, size_t nstr, int nspin)
{

    size_t istr;
    int ispin, srun;
    for (istr = 0; istr < nstr; istr++){
        srun = 0;
        for (ispin = 0; ispin < nspin; ispin++){
            if (1ULL << ispin & scstrs[istr]){ srun++; } else { srun--; }
            if (srun < 0){ mask[istr] = false; break; }
        }
    }

}

void FCICSFstrs2addr (int * addrs, uint64_t * strings, size_t nstr, int * gentable_ravel, int nspin, int twoS)
{

    /*  Example of a genealogical coupling table for 8 spins and s = 1 (triplet), counting from the final state
        back to the null state:

           28 28 19 10  4  1  .
            |  9  9  6  3  1  .
            |  |  3  3  2  1  .
            |  |  |  1  1  t  .
                        .  .  .
                           .  .
                              .

        n0 = 3 zero bits; n1 = 5 one bits -> 4x6 matrix. Going to the right is a 1 bit, going down is a 0 bit (bits sorted right to left).
        Notice how gen[i0,i1] = sum_(j0=i0)^n0 gen[j0,i1+1] (t == 1) except for the final column (gen[i0,n1] = 1).
        Example: 11001011: right twice, then down (instead of hitting 10 to the right), then right, then down twice (instead of hitting 3 or 2 to the right),
        then right twice. Address = 10 + 3 + 2 = 15.
        The next would be 01110011: right twice, then down twice (avoiding 10 and 6), then right three times, then down once (in the final column).
        Address = 10 + 6 + 0 = 16.
        Top left (0,0) is the null state (nspin = 0, s = 0).
    */
    size_t istr;
    int n0 = (nspin - twoS) / 2;
    int n1 = (nspin + twoS) / 2;
    int i0, i1;
    int ** gentable = malloc ((n0+1) * sizeof (int*));
    for (i0 = 0; i0 <= n0; i0++){
        gentable[i0] = & (gentable_ravel[i0*(n1+1)]);
    }
    int ispin;

    for (istr = 0; istr < nstr; istr++){
        addrs[istr] = 0;
        assert (1ULL & strings[istr]);
        i0 = 0, i1 = 0;
        for (ispin = 0; ispin < nspin; ispin++){
            if (1ULL << ispin & strings[istr]){  
                i1++;
            } else {
                if (i1 < n1) { addrs[istr] += gentable[i0][i1+1]; }
                i0++;
            }
        }
        assert (i0 == n0);
        assert (i1 == n1);
    }
    free (gentable);
}


void FCICSFaddrs2str (uint64_t * strings, int * addrs, size_t nstr, int * gentable_ravel, int nspin, int twoS)
{

    /*  Example of a genealogical coupling table for 8 spins and s = 1 (triplet), counting from the final state
        back to the null state:

           28 28 19 10  4  1  .
            |  9  9  6  3  1  .
            |  |  3  3  2  1  .
            |  |  |  1  1  t  .
                        .  .  .
                           .  .
                              .

        n0 = 3 zero bits; n1 = 5 one bits -> 4x6 matrix. Going to the right is a 1 bit, going down is a 0 bit (bits sorted right to left).
        Notice how gen[i0,i1] = sum_(j0=i0)^n0 gen[j0,i1+1] (t == 1) except for the final column (gen[i0,n1] = 1).
        Example: 11001011: right twice, then down (instead of hitting 10 to the right), then right, then down twice (instead of hitting 3 or 2 to the right),
        then right twice. Address = 10 + 3 + 2 = 15.
        The next would be 01110011: right twice, then down twice (avoiding 10 and 6), then right three times, then down once (in the final column).
        Address = 10 + 6 + 0 = 16.
    */
    size_t istr;
    int n0 = (nspin - twoS) / 2;
    int n1 = (nspin + twoS) / 2;
    int i0, i1;
    int ** gentable = malloc ((n0+1) * sizeof (int*));
    for (i0 = 0; i0 <= n0; i0++){
        gentable[i0] = &(gentable_ravel[i0*(n1+1)]);
    }
    int ispin, caddrs;

    for (istr = 0; istr < nstr; istr++){
        strings[istr] = 1ULL;
        caddrs = addrs[istr];
        i0 = 0, i1 = 0;
        for (ispin = 0; ispin < nspin; ispin++){
            if (i1 == n1){ break; }
            else if (gentable[i0][i1+1] <= caddrs){
                caddrs -= gentable[i0][i1+1];
                assert (i0 < i1);
                i0++;
            } else {
                strings[istr] |= 1ULL << ispin;
                i1++;
            }
            
        }
        assert (caddrs == 0);
    }
    free (gentable);
}

void FCICSFhdiag (double * hdiag, double * hdiag_det, double * eri, uint64_t * astrs, uint64_t * bstrs, unsigned int norb, size_t nconf, size_t ndet)
{

    size_t ndet_lt = ndet * (ndet+1) / 2;

#pragma omp parallel default(shared)
{

    size_t iconf, idetx, idety, idetconf;
    unsigned int iorb, nexc;
    uint64_t exc_str, somo_str, big_idx1, big_idx2, hdiag_idx_lt, hdiag_idx_ut;
    unsigned int exc[2];
    int sgn, esgn;

#pragma omp for schedule(static) 

    for (idetconf = 0; idetconf < nconf * ndet_lt; idetconf++){
        iconf = idetconf / ndet_lt;
        idety = idetconf % ndet_lt;
        for (idetx = 0; idetx < ndet; idetx++){
            if (idetx < idety){ idety -= idetx + 1; }
            else { break; }
        }
        // Careful with possible integer overflow
        hdiag_idx_lt = ndet;
        hdiag_idx_lt *= ndet;
        hdiag_idx_lt *= iconf;
        hdiag_idx_ut = hdiag_idx_lt;
        big_idx1 = ndet;
        big_idx1 *= idety;
        big_idx2 = ndet;
        big_idx2 *= idetx;
        hdiag_idx_lt += big_idx1;
        hdiag_idx_ut += big_idx2;
        hdiag_idx_lt += idetx;
        hdiag_idx_ut += idety;
        if (idetx == idety){ 
            hdiag[hdiag_idx_lt] = hdiag_det[(iconf*ndet)+idetx];
            continue;
        }
        // Fear of integer overflow is only reasonable for off-diagonal elements of a Hamiltonian matrix
        // It's not reasonable for anything else
        big_idx1 = (ndet*iconf) + idetx;
        big_idx2 = (ndet*iconf) + idety;
        exc_str  = astrs[big_idx1] ^ astrs[big_idx2];
        somo_str = astrs[big_idx1] ^ bstrs[big_idx1];
        nexc = 0; esgn = 1; sgn = -1;
        for (iorb = 0; iorb < norb; iorb++){
            if (somo_str & 1ULL << iorb){ esgn *= -1; }
            if (exc_str & 1ULL << iorb){
                if (nexc < 2){ exc[nexc] = iorb; }

                nexc++;
                if (nexc > 2){ break; }
                sgn *= esgn;
            }
        } 
        if (nexc > 2){ continue; }
        assert (nexc == 2);

        // Fear of integer overflow is only reasonable for off-diagonal elements of a Hamiltonian matrix
        // It's not reasonable for anything else
        big_idx1 = exc[0]*norb*norb*norb + exc[1]*norb*norb + exc[1]*norb + exc[0];
        hdiag[hdiag_idx_lt] = sgn * eri[big_idx1];
        hdiag[hdiag_idx_ut] = hdiag[hdiag_idx_lt];
    }

}
}

//RV 06/2026 : draft for norb>64 format
void FCICSFhdiag_occ (double * hdiag, double * hdiag_det, double * eri, unsigned int neleca, int * aoccs, unsigned int nelecb, int * boccs, unsigned int norb, size_t nconf, size_t ndet)
{

    size_t ndet_lt = ndet * (ndet+1) / 2;

#pragma omp parallel default(shared)
{
    size_t iconf, idetx, idety, idetconf;
    int nexc, exc[2];
    uint64_t big_idx1, big_idx2, hdiag_idx_lt, hdiag_idx_ut;
    int sgn, esgn;
    int *aocc1, *aocc2, *bocc;

#pragma omp for schedule(static) 

    for (idetconf = 0; idetconf < nconf * ndet_lt; idetconf++){
        iconf = idetconf / ndet_lt;
        idety = idetconf % ndet_lt;
        for (idetx = 0; idetx < ndet; idetx++){
            if (idetx < idety){ idety -= idetx + 1; }
            else { break; }
        }
        // Careful with possible integer overflow
        hdiag_idx_lt = ndet;
        hdiag_idx_lt *= ndet;
        hdiag_idx_lt *= iconf;
        hdiag_idx_ut = hdiag_idx_lt;
        big_idx1 = ndet;
        big_idx1 *= idety;
        big_idx2 = ndet;
        big_idx2 *= idetx;
        hdiag_idx_lt += big_idx1;
        hdiag_idx_ut += big_idx2;
        hdiag_idx_lt += idetx;
        hdiag_idx_ut += idety;
        if (idetx == idety){ 
            hdiag[hdiag_idx_lt] = hdiag_det[(iconf*ndet)+idetx];
            continue;
        }
        // Fear of integer overflow is only reasonable for off-diagonal elements of a Hamiltonian matrix
        // It's not reasonable for anything else
        big_idx1 = (ndet*iconf) + idetx;
        big_idx2 = (ndet*iconf) + idety;

        if(neleca == 0 || aoccs == NULL){aocc1 = NULL; aocc2 = NULL;
        } else{aocc1 = &aoccs[big_idx1 * neleca];aocc2 = &aoccs[big_idx2 * neleca];
        }
        if(nelecb == 0 || boccs == NULL){bocc = NULL;
        } else{bocc = &boccs[big_idx1 * nelecb];
        }

        nexc = 0,sgn = -1,esgn = 1;
        // excitation determination
        int i = 0, j = 0;
        while (i < neleca && j < neleca) {
            if (aocc1[i] < aocc2[j]) {
                if (nexc < 2) { exc[nexc] = aocc1[i]; }
                nexc++;
                i++;
            } else if (aocc2[j] < aocc1[i]) {
                if (nexc < 2) { exc[nexc] = aocc2[j]; }
                nexc++;
                j++;
            } else {
                i++;
                j++;
            }
            if (nexc > 2) { break; }
        }
        while (i < neleca && nexc <= 2) {
            if (nexc < 2) { exc[nexc] = aocc1[i]; }
                nexc++; i++;
            }
        while (j < neleca && nexc <= 2) {
            if (nexc < 2) { exc[nexc] = aocc2[j]; }
                nexc++; j++;
        }
        if (nexc > 2) { continue; }
        assert(nexc == 2);

        // Phase tracking
        int idx_a = 0, idx_b = 0, idx_e = 0;
        while (idx_a < neleca || idx_b < nelecb || idx_e < 2) {
            int next_orb = 1000000000; // Infinity placeholder
            if (idx_a < neleca) next_orb = aocc1[idx_a];
            if (idx_b < nelecb && bocc[idx_b] < next_orb) next_orb = bocc[idx_b];
            if (idx_e < 2      && exc[idx_e]   < next_orb) next_orb = exc[idx_e];
            if (next_orb > exc[1]) {
                break;
            }
            bool in_alpha = (idx_a < neleca && aocc1[idx_a] == next_orb);
            bool in_beta  = (idx_b < nelecb && bocc[idx_b] == next_orb);
            bool is_exc   = (idx_e < 2      && exc[idx_e]   == next_orb);
            
            if (in_alpha != in_beta) {
                esgn *= -1;
            }
            if (is_exc) {
                sgn *= esgn;
                idx_e++;
            }
            if (in_alpha) idx_a++;
            if (in_beta)  idx_b++;
        }

        big_idx1 = exc[0]*norb*norb*norb + exc[1]*norb*norb + exc[1]*norb + exc[0];
        hdiag[hdiag_idx_lt] = sgn * eri[big_idx1];
        hdiag[hdiag_idx_ut] = hdiag[hdiag_idx_lt];
    }
}
}

int FCIcre_des_sign_occ(int p, int q, int *occs, int nelec)
{
        int max_orb = (p > q) ? p : q;
        int min_orb = (p > q) ? q : p;
        int nocc = 0;
        for (int n = 0 ; n < nelec ; n++){
                if ( occs[n] > min_orb && occs[n] < max_orb) nocc++;
        }
        if (nocc % 2){
            return -1;
        }else {
            return 1;
        }
}

void occ_cre_des(int *occ,int *occ_in, int cre,int des,int nelec)
{
    occ[0] = -1;
    bool created = false;
    int ni = 1;
    for (int i = 0 ; i < nelec ; i++){
        if (occ_in[i] == des){continue;
        } else if(occ_in[i] > cre && !created){occ[ni] = cre; ni++;created=true ;occ[ni] = occ_in[i];ni++;
        } else {occ[ni] = occ_in[i];ni++;}
    }
    if(!created){occ[nelec-1] = cre;}
}

void FCIpspace_h0tril_occ(double *h0, double *h1e_a, double *h1e_b,
                          double *g2e_aa, double *g2e_ab, double *g2e_bb,
                          int *aoccs, int *boccs,
                          int neleca, int nelecb,
                          int norb, int np)
{
        const int d2 = norb * norb;
        const int d3 = norb * norb * norb;
#pragma omp parallel
{
        int i, j, k, pi, pj, pk, pl;
        int n1da, n1db,n1dai, n1dbi,n1daj, n1dbj;
        int acre_des[neleca],bcre_des[nelecb],aexci[2],bexci[2],aexcj[2],bexcj[2];
        int *aocc1,*aocc2,*bocc1,*bocc2;
        double tmp;
#pragma omp for schedule(dynamic)
        for (i = 0; i < np; i++) {
        for (j = 0; j < i; j++) {
                /* 64-bit integer to avoid integer overflow */
                uint64_t idx = (uint64_t)i * (uint64_t)np + (uint64_t)j;

                if(neleca == 0 || aoccs == NULL){aocc1 = NULL; aocc2 = NULL;
                } else{aocc1 = &aoccs[i * neleca];aocc2 = &aoccs[j * neleca];
                }
                if(nelecb == 0 || boccs == NULL){bocc1 = NULL; bocc2 = NULL;
                } else{bocc1 = &boccs[i * nelecb];bocc2 = &boccs[j * nelecb];
                }

                n1da = 0, n1dai = 0, n1daj = 0;
                int ni = 0,nj = 0;
                while (ni < neleca && nj < neleca) {
                    if (aocc1[ni] < aocc2[nj]) {if (n1dai < 2) { aexci[n1dai] = aocc1[ni]; } n1dai++;ni++;
                    } else if (aocc2[nj] < aocc1[ni]) {if (n1daj < 2) { aexcj[n1daj] = aocc2[nj]; }n1daj++;nj++;
                    } else {ni++;nj++;
                    } if (n1dai + n1daj > 4) { break; }
                }
                while (ni < neleca && n1dai <= 2) {if (n1dai < 2) { aexci[n1dai] = aocc1[ni]; }n1dai++; ni++;}
                while (nj < neleca && n1daj <= 2) {if (n1daj < 2) { aexcj[n1daj] = aocc2[nj]; }n1daj++; nj++;}   
                n1da = n1dai + n1daj;

                n1db = 0, n1dbi = 0, n1dbj = 0;
                ni = 0,nj = 0;
                while (ni < nelecb && nj < nelecb) {
                    if (bocc1[ni] < bocc2[nj]) {if (n1dbi < 2) { bexci[n1dbi] = bocc1[ni]; } n1dbi++;ni++;
                    } else if (bocc2[nj] < bocc1[ni]) {if (n1dbj < 2) { bexcj[n1dbj] = bocc2[nj]; }n1dbj++;nj++;
                    } else {ni++;nj++;
                    } if (n1dbi + n1dbj > 4) { break; }
                }
                while (ni < nelecb && n1dbi <= 2) {if(n1dbi<2){ bexci[n1dbi] = bocc1[ni];} n1dbi++; ni++;}
                while (nj < nelecb && n1dbj <= 2) {if(n1dbj<2){ bexcj[n1dbj] = bocc2[nj];} n1dbj++; nj++;}  
                n1db = n1dbi + n1dbj;

                switch (n1da) {
                case 0: switch (n1db) {
                        case 2:
                        pi = bexci[0];
                        pj = bexcj[0];
                        tmp = h1e_b[pi*norb+pj];
                        for (ni = 0; ni < neleca; ni++) {
                            k = aocc1[ni];
                            tmp += g2e_ab[pi*norb+pj+k*d3+k*d2];
                        }
                        for (ni = 0; ni < nelecb; ni++) {
                            k = bocc1[ni];
                            tmp += g2e_bb[pi*d3+pj*d2+k*norb+k]
                                - g2e_bb[pi*d3+k*d2+k*norb+pj];
                        }
                        if (FCIcre_des_sign_occ(pi, pj, bocc2 ,nelecb) > 0) {
                                h0[idx] = tmp;
                        } else {
                                h0[idx] = -tmp;
                        } break;

                        case 4:
                        pi = bexci[0];
                        pj = bexcj[0];
                        pk = bexci[1];
                        pl = bexcj[1];
                        occ_cre_des(bcre_des,bocc2,pi,pj,nelecb);
                        if (FCIcre_des_sign_occ(pi, pj, bocc2, nelecb)
                           *FCIcre_des_sign_occ(pk, pl, bcre_des, nelecb) > 0) {
                                h0[idx] = g2e_bb[pi*d3+pj*d2+pk*norb+pl]
                                           - g2e_bb[pi*d3+pl*d2+pk*norb+pj];
                        } else {
                                h0[idx] =-g2e_bb[pi*d3+pj*d2+pk*norb+pl]
                                           + g2e_bb[pi*d3+pl*d2+pk*norb+pj];
                        } } break;
                case 2: switch (n1db) {
                        case 0:
                        pi = aexci[0];
                        pj = aexcj[0];
                        tmp = h1e_a[pi*norb+pj];
                        for (ni = 0; ni < nelecb; ni++) {
                            k = bocc1[ni];
                            tmp += g2e_ab[pi*d3+pj*d2+k*norb+k];
                        }
                        for (ni = 0; ni < neleca; ni++) {
                            k = aocc1[ni];
                            tmp += g2e_aa[pi*d3+pj*d2+k*norb+k]
                                - g2e_aa[pi*d3+k*d2+k*norb+pj];
                        }
                        if (FCIcre_des_sign_occ(pi, pj, aocc2, neleca) > 0) {
                                h0[idx] = tmp;
                        } else {
                                h0[idx] = -tmp;
                        } break;
                        case 2:
                        pi = aexci[0];
                        pj = aexcj[0];
                        pk = bexci[0];
                        pl = bexcj[0];
                        if (FCIcre_des_sign_occ(pi, pj, aocc2, neleca)
                           *FCIcre_des_sign_occ(pk, pl, bocc2, nelecb) > 0) {
                                h0[idx] = g2e_ab[pi*d3+pj*d2+pk*norb+pl];
                        } else {
                                h0[idx] =-g2e_ab[pi*d3+pj*d2+pk*norb+pl];
                        } } break;
                case 4: switch (n1db) {
                        case 0:
                        pi = aexci[0];
                        pj = aexcj[0];
                        pk = aexci[1];
                        pl = aexcj[1];
                        occ_cre_des(acre_des,aocc2,pi,pj,neleca);
                        if (FCIcre_des_sign_occ(pi, pj, aocc2, neleca)
                           *FCIcre_des_sign_occ(pk, pl, acre_des, neleca) > 0) {
                                h0[idx] = g2e_aa[pi*d3+pj*d2+pk*norb+pl]
                                           - g2e_aa[pi*d3+pl*d2+pk*norb+pj];
                        } else {
                                h0[idx] =-g2e_aa[pi*d3+pj*d2+pk*norb+pl]
                                           + g2e_aa[pi*d3+pl*d2+pk*norb+pj];
                        }
                        } break;
                }
        } }
}
}