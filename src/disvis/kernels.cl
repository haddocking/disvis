#define SQUARE(a) ((a) * (a))

// To be defined on compile time
#define INTERACTION_CUTOFF $interaction_cutoff
#define NRESTRAINTS $nrestraints
#define SHAPE_X $shape_x
#define SHAPE_Y $shape_y
#define SHAPE_Z $shape_z
#define LLENGTH $llength
#define NRECEPTOR_COOR $nreceptor_coor
#define NLIGAND_COOR $nligand_coor

#define INTERACTION_CUTOFF2 (INTERACTION_CUTOFF * INTERACTION_CUTOFF)
#define NRESTRAINTS2 (NRESTRAINTS * NRESTRAINTS)
#define NWORDS ((NRESTRAINTS + 31) / 32)
#define LLENGTH2 (LLENGTH * LLENGTH)
#define SLICE ((SHAPE_X * SHAPE_Y))
#define SIZE ((SHAPE_Z * SLICE))
#define TOTAL_COOR ((NRECEPTOR_COOR + NLIGAND_COOR))

kernel
void rotate_points3d(global float3 *in, int npoints, float16 rotmat, global float3 *out) 
{
    int id = get_global_id(0);
    int stride = get_global_size(0);
    int i;

    for (i = id; i < npoints; i += stride) {
        out[i].s0 = rotmat.s0 * in[i].s0 +
                    rotmat.s1 * in[i].s1 +
                    rotmat.s2 * in[i].s2;
        out[i].s1 = rotmat.s3 * in[i].s0 +
                    rotmat.s4 * in[i].s1 +
                    rotmat.s5 * in[i].s2;
        out[i].s2 = rotmat.s6 * in[i].s0 +
                    rotmat.s7 * in[i].s1 +
                    rotmat.s8 * in[i].s2;
    }
}


kernel
void rotate_grid3d(
        global float *grid, float16 rotmat, global float *out
        )
{
    // Rotate grid around the origin. Only grid points within LLENGTH of the
    // origin are rotated. Nearest neighbour interpolation.

    int z, y, x;
    float3 dist2, coor_z, coor_zy;
    int3 index, coor;

    int zid = get_global_id(0);
    int yid = get_global_id(1);
    int xid = get_global_id(2);
    int zstride = get_global_size(0);
    int ystride = get_global_size(1);
    int xstride = get_global_size(2);

    for (z = zid - LLENGTH; z <= LLENGTH; z += zstride) {
        dist2.s2 = SQUARE(z);
        coor_z.s0 = rotmat.s6 * z;
        coor_z.s1 = rotmat.s7 * z;
        coor_z.s2 = rotmat.s8 * z;

        index.s0 = z * SLICE;
        // Wraparound the z-coordinate
        if (z < 0)
            index.s0 += SIZE;

        for (y = yid - LLENGTH; y <= LLENGTH; y += ystride) {
            dist2.s1 = SQUARE(y) + dist2.s2;
            coor_zy.s0 = rotmat.s3 * y + coor_z.s0;
            coor_zy.s1 = rotmat.s4 * y + coor_z.s1;
            coor_zy.s2 = rotmat.s5 * y + coor_z.s2;

            index.s1 = index.s0 + y * SHAPE_X;
            // Wraparound the y-coordinate
            if (y < 0)
                index.s1 += SLICE;

            for (x = xid - LLENGTH; x <= LLENGTH; x += xstride) {
                dist2.s0 = SQUARE(x) + dist2.s1;
                if (dist2.s0 > LLENGTH2)
                    continue;

                coor.s0 = (int) round(rotmat.s0 * x + coor_zy.s0);
                coor.s1 = (int) round(rotmat.s1 * x + coor_zy.s1);
                coor.s2 = (int) round(rotmat.s2 * x + coor_zy.s2);

                index.s2 = index.s1 + x;
                if (x < 0)
                    index.s2 += SHAPE_X;
                if (coor.s0 < 0)
                    coor.s0 += SHAPE_X;
                if (coor.s1 < 0)
                    coor.s1 += SHAPE_Y;
                if (coor.s2 < 0)
                    coor.s2 += SHAPE_Z;

                out[index.s2] = grid[coor.s2 * SLICE +
                                     coor.s1 * SHAPE_X +
                                     coor.s0
                                     ];
            }
        }
    }
}


kernel
void restraint_bits(
        global float4 *gcenter, global int *grestraint,
        global float *gmindis, global float *gmaxdis, int ncenters,
        global uint *bits
        )
{
    // Set bit n of the restraint words of every translation within the
    // distance range of a center of restraint n; grestraint[p] is the
    // restraint of center p, and word w of translation i is
    // bits[w * SIZE + i]. A work group handles one center at a time, and its
    // work items the grid points of the box around the center.

    int p, n, k, x, y, z, xmin, ymin, zmin, nx, ny, nz;
    float dis2_z, dis2_zy, dis2_zyx, maxdis, mindis2, maxdis2;
    float4 center;

    for (p = get_group_id(0); p < ncenters; p += get_num_groups(0)) {
        n = grestraint[p];
        center = gcenter[p];
        maxdis = gmaxdis[n];
        mindis2 = SQUARE(gmindis[n]);
        maxdis2 = SQUARE(maxdis);
        xmin = max((int) ceil(center.s0 - maxdis), 0);
        ymin = max((int) ceil(center.s1 - maxdis), 0);
        zmin = max((int) ceil(center.s2 - maxdis), 0);
        nx = min((int) floor(center.s0 + maxdis), SHAPE_X - 1) - xmin + 1;
        ny = min((int) floor(center.s1 + maxdis), SHAPE_Y - 1) - ymin + 1;
        nz = min((int) floor(center.s2 + maxdis), SHAPE_Z - 1) - zmin + 1;
        if ((nx <= 0) || (ny <= 0) || (nz <= 0))
            continue;
        for (k = get_local_id(0); k < nx * ny * nz; k += get_local_size(0)) {
            z = zmin + k / (nx * ny);
            y = ymin + (k / nx) % ny;
            x = xmin + k % nx;
            dis2_z = SQUARE(z - center.s2);
            dis2_zy = SQUARE(y - center.s1) + dis2_z;
            dis2_zyx = dis2_zy + SQUARE(x - center.s0);
            if ((dis2_zyx <= maxdis2) && (dis2_zyx >= mindis2))
                atomic_or(bits + (n / 32) * SIZE + z * SLICE + y * SHAPE_X + x,
                          1u << (n % 32));
        }
    }
}


kernel
void reduced_interspace(
        global uint *bits, global int *interspace, global int *red_interspace,
        global int *access_interspace
        )
{
    // For every translation in the interaction space, count the distance
    // restraints it is consistent with (the bits set by restraint_bits),
    // and update the accessible interaction space, the highest count at
    // every translation.

    size_t gid = get_global_id(0);
    size_t gstride = get_global_size(0);

    int i, w, count;

    for (i = gid; i < SIZE; i += gstride) {
        if (interspace[i] == 0) {
            red_interspace[i] = 0;
            continue;
        }
        count = 0;
        for (w = 0; w < NWORDS; w++)
            count += popcount(bits[w * SIZE + i]);
        red_interspace[i] = count;
        if (count > access_interspace[i])
            access_interspace[i] = count;
    }
}


kernel
void histogram(
        global int *data, global int *hist
        )
{
    // Count complexes consistent with N restraints

    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t lstride = get_local_size(0);
    size_t gstride = get_global_size(0);
    int i, value;

    local int local_hist[NRESTRAINTS];

    // Set the local histogram to zero
    for (i = lid; i < NRESTRAINTS; i += lstride)
        local_hist[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Fill in the local histogram. Only count where data > 0.
    for (i = gid; i < SIZE; i += gstride) {
        value = data[i];
        if (value > 0)
            atomic_inc(local_hist + value - 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Move the local histogram to global memory
    for (i = lid; i < NRESTRAINTS; i += lstride) {
        atomic_add(hist + i, local_hist[i]);
    }
}


kernel
void count_violations(
        global uint *bits, global int *interspace, global int *viol
        )
{
    // Count how often each restraint is violated in complexes consistent with
    // N restraints: restraint i is violated where restraint_bits did not set
    // its bit.

    int i, x, y, z;
    int ind_z, ind_zy, consistent, offset;
    local int local_viol[NRESTRAINTS2];

    size_t zid = get_global_id(0);
    size_t yid = get_global_id(1);
    size_t xid = get_global_id(2);
    size_t zstride = get_global_size(0);
    size_t ystride = get_global_size(1);
    size_t xstride = get_global_size(2);
    size_t lid = get_local_id(0);
    size_t lstride = get_local_size(0);

    // Prepare local memory
    for (i = lid; i < NRESTRAINTS2; i += lstride)
        local_viol[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over the accessible interaction space and count the violations
    for (z = zid; z < SHAPE_Z; z += zstride) {
        ind_z = z * SLICE;
        for (y = yid; y < SHAPE_Y; y += ystride) {
            ind_zy = y * SHAPE_X + ind_z;
            for (x = xid; x < SHAPE_X; x += xstride) {
                consistent = interspace[ind_zy + x];
                // Do not process translations where there are no consistent
                // restraints, or when all restraints are consistent. This is redundant.
                if ((consistent == 0))// || (consistent == NRESTRAINTS))
                    continue;
                offset = (consistent - 1) * NRESTRAINTS;
                for (i = 0; i < NRESTRAINTS; i++) {
                    if (!((bits[(i / 32) * SIZE + ind_zy + x] >> (i % 32)) & 1))
                        atomic_inc(local_viol + offset + i);
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Merge the local histograms in global memory
    for (i = lid; i < NRESTRAINTS2; i += lstride)
        atomic_add(viol + i, local_viol[i]);
}


kernel
void count_interactions(
        global float3 *fixed_coor, global float3 *scanning_coor,
        global int *inter_space, int nconsistent, global int *hist
        )
{
    // Count the number of interactions each residue makes for complexes
    // consistent with nrestraints.

    global int *offset;
    local float3 l_fixed_coor[NRECEPTOR_COOR];
    int x, y, z, ind_z, ind_zy, i, j;
    float dist2;
    float3 coor, moved_coor;

    size_t zid = get_global_id(0);
    size_t yid = get_global_id(1);
    size_t xid = get_global_id(2);
    size_t zstride = get_global_size(0);
    size_t ystride = get_global_size(1);
    size_t xstride = get_global_size(2);

    size_t lid = get_local_id(0);
    size_t lstride = get_local_size(0);

    // Move fixed coor to local memory
    for (i = lid; i < NRECEPTOR_COOR; i += lstride)
        l_fixed_coor[i] = fixed_coor[i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over the interaction space, and count interactions for
    // conformations that are consistent with nrestraints.
    for (z = zid; z < SHAPE_Z; z += zstride) {
        ind_z = z * SLICE;
        for (y = yid; y < SHAPE_Y; y += ystride) {
            ind_zy = y * SHAPE_X + ind_z;
            for (x = xid; x < SHAPE_X; x += xstride) {

                // Only investigate conformations consistent with nconsistent
                if (inter_space[ind_zy + x] != nconsistent)
                    continue;

                for (i = 0; i < NLIGAND_COOR; i++) {
                    offset = hist + i * NRECEPTOR_COOR;
                    moved_coor = scanning_coor[i] + (float3) (x, y, z);
                    for (j = 0; j < NRECEPTOR_COOR; j++) {
                        dist2 = SQUARE(moved_coor.s0 - l_fixed_coor[j].s0) +
                                SQUARE(moved_coor.s1 - l_fixed_coor[j].s1) +
                                SQUARE(moved_coor.s2 - l_fixed_coor[j].s2);
                        if (dist2 <= INTERACTION_CUTOFF2)
                            atomic_inc(offset + j);
                    }
                }
            }
        }
    }
}
