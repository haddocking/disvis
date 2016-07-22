#define SQUARE(a) ((a) * (a))
#define INTERACTION_CUTOFF 10
#define INTERACTION_CUTOFF2 100


kernel
void rotate_points(global float4 *in, uint in_size, float16 rotmat, global float4 *out)
{

    uint i;

    for (i = get_global_id(0); i < in_size; i += get_global_size(0)) {
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
void count_interactions(
        global int *inter_space, uint4 inter_space_shape,
        global float4 *fixed_coor, uint fixed_coor_size,
        global float4 *scanning_coor, uint scanning_coor_size,
        global uint *inter_hist,
        local uint *loc_inter_hist,
        int nrestraints,
        )
{
    // Count the number of interactions each residue makes for complexes
    // consistent with nrestraints.

    size_t lid = get_local_id(0);
    size_t lstride = get_local_size(0);

    int nz, ny, nx, i, j;
    int2 inter_ind;
    float dist2;

    // Set local histogram to 0
    if (lid < (scanning_coor_size + fixed_coor_size)) {
        for (i = lid; i < (scanning_coor_size + fixed_coor_size); i += lstride)
            loc_inter_hist[i] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over the interaction space, and count interactions for
    // conformations that are consistent with nrestraints.
    for (nz = get_global_id(0); nz < inter_space_shape.s0; nz += get_global_size(0)) {
        inter_ind.s0 = nz * inter_space_shape.s0;
        for (ny = get_global_id(1); ny < inter_space_shape.s1; ny += get_global_size(1)) {
            inter_ind.s1 = inter_ind.s0 + ny * inter_space_shape.s2;
            for (nx = get_global_id(2); nx < inter_space_shape.s2; nx += get_global_size(2)) {

                // Only investigate conformations consistent with nrestraints
                if (inter_space[inter_ind.s1 + nx] != nrestraints)
                    continue;

                // Calculate the number of interactions for each residue
                for (i = 0; i < scanning_coor_size; i++) {
                    coor.s0 = scanning_coor[i].s0 + nx;
                    coor.s1 = scanning_coor[i].s1 + ny;
                    coor.s2 = scanning_coor[i].s2 + nz;
                    for (j = 0; j < fixed_coor_size; j++) {
                        dist2 = SQUARE(coor.s0 - fixed_coor[j].s0) +
                                SQUARE(coor.s1 - fixed_coor[j].s1) +
                                SQUARE(coor.s2 - fixed_coor[j].s2);
                        if (dist2 <= INTERACTION_CUTOFF2) {
                            // Increase the counter using atomics
                            atomic_inc(loc_inter_hist + j);
                            atomic_inc(loc_inter_hist + i + fixed_coor_size);
                        }
                    }
                }
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Combine the local histograms into the global memory
    if (lid < (scanning_coor_size + fixed_coor_size)) {
        offset = (scanning_coor_size + fixed_coor_size) * get_group_id(0);
        for (i = lid; i < scanning_coor_size; i += lstride)
            inter_hist[offset + i] += loc_inter_hist[i];
    }
}


__kernel
void count_violations(global float8 *restraints,
                      float16 rotmat,
                      global int *access_interspace,
                      global float *viol_counter,
                      local float *loc_viol,
                      local float4 *restraints_center,
                      local float *mindist2,
                      local float *maxdist2,
                      uint nrestraints, uint4 shape, float weight)
{
    uint id = get_global_id(0);
    uint stride = get_global_size(0);
    uint lid = get_local_id(0);
    uint slice = shape.s2 * shape.s1;

    //set loc_viol to 0
    uint nrestraints2 = nrestraints*nrestraints;
    uint ind_z = lid*nrestraints2;
    for (uint i = 0; i < nrestraints2; i++)
        loc_viol[i + ind_z] = 0;

    // calculate the center of the restraints
    if (lid < nrestraints) {
        restraints_center[lid].s0 = restraints[lid].s0 -
                                   (rotmat.s0*restraints[lid].s3 +
                                    rotmat.s1*restraints[lid].s4 +
                                    rotmat.s2*restraints[lid].s5);
        restraints_center[lid].s1 = restraints[lid].s1 -
                                   (rotmat.s3*restraints[lid].s3 +
                                    rotmat.s4*restraints[lid].s4 +
                                    rotmat.s5*restraints[lid].s5);
        restraints_center[lid].s2 = restraints[lid].s2 -
                                   (rotmat.s6*restraints[lid].s3 +
                                    rotmat.s7*restraints[lid].s4 +
                                    rotmat.s8*restraints[lid].s5);
        mindist2[lid] = SQUARE(restraints[lid].s6);
        maxdist2[lid] = SQUARE(restraints[lid].s7);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // loop over the accessable interaction space
    for (uint i = id; i < shape.s3; i += stride) {
        int consistent = access_interspace[i];
        if ((consistent == 0) || (consistent == nrestraints))
            continue;
        consistent--;

        // get x, y, z map coordinate
        uint z = i/slice;
        uint y = (i - slice*z)/shape.s2;
        uint x = i - slice*z - y*shape.s2;

        // check which restraints are violated for a certain number
        // of consistent restraints
        for (uint j = 0; j < nrestraints; j++){
            float dist2 = SQUARE(x - restraints_center[j].s0) +
                          SQUARE(y - restraints_center[j].s1) +
                          SQUARE(z - restraints_center[j].s2);
            if ((dist2 < mindist2[j]) || (dist2 > maxdist2[j])){
                uint ind = ind_z + nrestraints*consistent + j;
                loc_viol[ind] += weight;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // merge the work item subhistograms into work group histogram
    if (lid < nrestraints2) {
        uint gind_z = get_group_id(0)*nrestraints2;
        for (uint i = lid; i < nrestraints2; i += get_local_size(0)){

            float sum = 0;
            for (uint j = 0; j < get_local_size(0); j++)
                sum += loc_viol[nrestraints2*j + i];

            viol_counter[gind_z + i] += sum;
        }
    }
}


kernel
void histogram(global int *data,
               global float *subhists,
               local int *local_hist,
               uint nrestraints,
               float weight,
               uint size)
{
    size_t lid = get_local_id(0);
    size_t lsize = get_local_size(0);
    size_t gsize = get_global_size(0);
    size_t groupid = get_group_id(0);
    int i, j, loffset, goffset, sum;

    // Set the local histogram to zero
    // Each local workitem has its own array of nrestraints to fill
    for (i = 0; i < nrestraints; i++)
        local_hist[lid + i * lsize] = 0;

    // Fill in the local histogram
    for (i = get_global_id(0); i < size; i += gsize)
        local_hist[lid + data[i] * lsize]++;

    barrier(CLK_LOCAL_MEM_FENCE);

    // Move the local histogram to global memory
    if (lid < nrestraints) {
        goffset = groupid * nrestraints;
        for (j = lid; j < nrestraints; j += lsize) {

            // Sum all complexes consistent with j restraints
            sum = 0;
            loffset = lsize * lid;
            for (i = 0; i < lsize; i++)
                sum += local_hist[loffset + i];

            subhists[goffset + j] += weight * sum;
        }
    }
}


kernel
void rotate_image3d(sampler_t sampler,
                    read_only image3d_t image,
                    float16 rotmat,
                   __global float *out,
                   float4 center,
                   int4 shape)
{
    int id = get_global_id(0);
    int stride = get_global_size(0);

    int x, y, z, slice;
    float xrot, yrot, zrot;
    float4 coordinate, weight;

    slice = shape.s2*shape.s1;
    float OFFSET = 0.5f;

    int i;
    for (i = id; i < shape.s3; i += stride) {

        z = i/slice;
        y = (i - z*slice)/shape.s2;
        x = i - z*slice - y*shape.s2;

        if (x >= 0.5*shape.s2)
            x -= shape.s2;
        if (y >= 0.5*shape.s1)
            y -= shape.s1;
        if (z >= 0.5*shape.s0)
            z -= shape.s0;

        xrot = rotmat.s0*x + rotmat.s1*y + rotmat.s2*z;
        yrot = rotmat.s3*x + rotmat.s4*y + rotmat.s5*z;
        zrot = rotmat.s6*x + rotmat.s7*y + rotmat.s8*z;

        xrot += OFFSET + center.s0;
        yrot += OFFSET + center.s1;
        zrot += OFFSET + center.s2;

        coordinate = (float4) (xrot, yrot, zrot, 0);
        weight = read_imagef(image, sampler, coordinate);

        out[i] = weight.s0;
    }
}


__kernel
void distance_restraint(__global float8 *restraints,
                       float16 rotmat,
                       __global int *restspace,
                       int4 shape, int nrestraints)
{

    uint zid = get_global_id(0);
    uint yid = get_global_id(1);
    uint xid = get_global_id(2);

    uint zstride = get_global_size(0);
    uint ystride = get_global_size(1);
    uint xstride = get_global_size(2);

    uint i, ix, iy, iz;
    uint z_ind, yz_ind, xyz_ind;
    float xligand, yligand, zligand;
    float xcenter, ycenter, zcenter, mindis2, maxdis2, z_dis2, yz_dis2, xyz_dis2;

    uint slice = shape.s2 * shape.s1;

    for (i = 0; i < nrestraints; i++){

         // determine the center of the point that will be dilated
         xligand = rotmat.s0 * restraints[i].s3 + 
                   rotmat.s1 * restraints[i].s4 + 
                   rotmat.s2 * restraints[i].s5;
         yligand = rotmat.s3 * restraints[i].s3 + 
                   rotmat.s4 * restraints[i].s4 + 
                   rotmat.s5 * restraints[i].s5;
         zligand = rotmat.s6 * restraints[i].s3 + 
                   rotmat.s7 * restraints[i].s4 + 
                   rotmat.s8 * restraints[i].s5;

         xcenter = restraints[i].s0 - xligand;
         ycenter = restraints[i].s1 - yligand;
         zcenter = restraints[i].s2 - zligand;

         mindis2 = SQUARE(restraints[i].s6);
         maxdis2 = SQUARE(restraints[i].s7);

         // calculate the distance of every voxel to the determined center
         for (iz = zid; iz < shape.s0; iz += zstride){

             z_dis2 = SQUARE(iz - zcenter);

             z_ind = iz * slice;

             for (iy = yid; iy < shape.s1; iy += ystride){
                 yz_dis2 = SQUARE(iy - ycenter) + z_dis2;

                 yz_ind = z_ind + iy*shape.s2;

                 for (ix = xid; ix < shape.s2; ix += xstride){

                     xyz_dis2 = SQUARE(ix - xcenter) + yz_dis2;

                     if ((xyz_dis2 <= maxdis2) && (xyz_dis2 >= mindis2)){
                         xyz_ind = ix + yz_ind;
                         restspace[xyz_ind] += 1;
                     }
                 }
             }
         }
    }
}
