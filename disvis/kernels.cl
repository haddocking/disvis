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
void dilate_point_add(
        global float4 *gcenter, global float *gmindis, global float *gmaxdis,
        global int *restspace
        )
{
    // Determine the restraint-consistent space.

    int zid = get_global_id(0);
    int yid = get_global_id(1);
    int xid = get_global_id(2);

    int zstride = get_global_size(0);
    int ystride = get_global_size(1);
    int xstride = get_global_size(2);

    int n, x, y, z, ind_z, ind_zy;
    int xmin, xmax, ymin, ymax, zmin, zmax;
    float dis2_z, dis2_zy, dis2_zyx, mindis, maxdis, mindis2, maxdis2;
    float4 center;

    for (n = 0; n < NRESTRAINTS; n++) {

        center = gcenter[n];
        maxdis = gmaxdis[n];
        mindis = gmindis[n];

        xmin = max((int) floor(center.s0 - maxdis), 0);
        xmax = min((int) ceil(center.s0 + maxdis), SHAPE_X - 1);
        ymin = max((int) floor(center.s1 - maxdis), 0);
        ymax = min((int) ceil(center.s1 + maxdis), SHAPE_Y - 1);
        zmin = max((int) floor(center.s2 - maxdis), 0);
        zmax = min((int) ceil(center.s2 + maxdis), SHAPE_Z - 1);
        mindis2 = SQUARE(mindis);
        maxdis2 = SQUARE(maxdis);

        for (z = zmin + zid; z <= zmax; z += zstride) {
            ind_z = z * SLICE;
            dis2_z = SQUARE(z - center.s2);
            for (y = ymin + yid; y <= ymax; y += ystride) {
                ind_zy = y * SHAPE_X + ind_z;
                dis2_zy = SQUARE(y - center.s1) + dis2_z;
                for (x = xmin + xid; x <= xmax; x += xstride) {
                    dis2_zyx = dis2_zy + SQUARE(x - center.s0);
                    if ((dis2_zyx <= maxdis2) && (dis2_zyx >= mindis2))
                        atomic_inc(restspace + ind_zy + x);
                }
            }
        }
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
        global float3 *center, global float *mindis2, global float *maxdis2,
        global int *interspace, global int *viol
        )
{
    // Count how often each restraint is violated in complexes consistent with
    // N restraints.

    int i, x, y, z;
    int ind_z, ind_zy, consistent, offset;
    float dist2;
    local int local_viol[NRESTRAINTS2];
    local float3 local_center[NRESTRAINTS];
    local float local_mindis2[NRESTRAINTS];
    local float local_maxdis2[NRESTRAINTS];

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
    // Move the centers and distances to local memory
    for (i = lid; i < NRESTRAINTS; i += lstride) {
            local_center[i] = center[i];
            local_maxdis2[i] = maxdis2[i];
            local_mindis2[i] = mindis2[i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over the accessable interaction space and count the violations
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
                    dist2 = SQUARE(x - local_center[i].s0) + 
                            SQUARE(y - local_center[i].s1) +
                            SQUARE(z - local_center[i].s2);
                    if ((dist2 < local_mindis2[i]) || (dist2 > local_maxdis2[i]))
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
