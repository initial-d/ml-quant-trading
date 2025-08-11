





awk '$3>0.008 {$4=log($4+1-0.00125); d=int(100*$4); if (d<0) d=-d; A[d]+=$4;B[d]+=1} END {w=0;for(i in A) w+=A[i];for (i in A) print i,A[i],B[i],A[i]/B[i],A[i]/w}' | sort -k1 -nr
#awk '$3>0.006 {$4=l$4-0.00125; d=int(100*$4); if (d<0) d=-d; A[d]+=$4;B[d]+=1} END {w=0;w1=0;for(i in A) if(a[i]>=0)w+=A[i];else w1=A[i];for (i in A) print i,A[i],B[i],A[i]/B[i],A[i]/w}' | sort -k1 -nr
