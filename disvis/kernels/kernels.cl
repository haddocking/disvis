#define UMUL(a, b) ((a) * (b))
#define UMAD(a, b, c) (UMUL((a), (b)) + (c))


__kernel
void count_violations(__global float8 *restraints,
                      float16 rotmat,
                      __global int *access_interspace,
                      __global float *viol_counter,
                      __local float *loc_viol,
                      __local float4 *restraints_center,
                      __local float *mindist2,
                      __local float *maxdist2,
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
        mindist2[lid] = pown(restraints[lid].s6, 2);
        maxdist2[lid] = pown(restraints[lid].s7, 2);
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
            float dist2 = pown(x - restraints_center[j].s0, 2) +
                          pown(y - restraints_center[j].s1, 2) +
                          pown(z - restraints_center[j].s2, 2);
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


__kernel
void copy_partial(__global float *part, __global float *full, int4 part_size, int4 full_size)
{
    uint z, y, x, full_ind;
    uint part_slice = part_size.s2*part_size.s1;
    uint full_slice = full_size.s2*full_size.s1;

    for (uint i = get_global_id(0); i < part_size.s3; i += get_global_size(0)) {
        z = i/part_slice;
        y = (i - z*part_slice)/part_size.s2;
        x = i - z*part_slice - y*part_size.s2;

        full_ind = z * full_slice + y * full_size.s2 + x;
        full[full_ind] = part[i];
    }
}


__kernel
void histogram(__global int *data,
                 __global float *subhists,
                 __local float *local_hist,
                 uint nrestraints,
                 float weight,
                 uint size)
{
    uint lid = get_local_id(0);

    for (uint i = 0; i < nrestraints; i++)
        local_hist[lid + i * get_local_size(0)] = 0;

    for (uint pos = get_global_id(0); pos < size; pos += get_global_size(0))
        local_hist[lid + data[pos] * get_local_size(0)] += weight;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < nrestraints) {
        for (uint j =  lid; j < nrestraints; j += get_local_size(0)) {

            float sum = 0;
            int pos = get_local_size(0) * lid;

            for (uint i = 0; i < get_local_size(0); i++)
                sum += local_hist[pos + i];

            subhists[get_group_id(0)*nrestraints + j] += sum;
        }
    }
}

__kernel
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
         xligand = rotmat.s0 * restraints[i].s3 + rotmat.s1 * restraints[i].s4 + rotmat.s2 * restraints[i].s5;
         yligand = rotmat.s3 * restraints[i].s3 + rotmat.s4 * restraints[i].s4 + rotmat.s5 * restraints[i].s5;
         zligand = rotmat.s6 * restraints[i].s3 + rotmat.s7 * restraints[i].s4 + rotmat.s8 * restraints[i].s5;

         xcenter = restraints[i].s0 - xligand;
         ycenter = restraints[i].s1 - yligand;
         zcenter = restraints[i].s2 - zligand;

         mindis2 = pown(restraints[i].s6, 2);
         maxdis2 = pown(restraints[i].s7, 2);

         // calculate the distance of every voxel to the determined center
         for (iz = zid; iz < shape.s0; iz += zstride){

             z_dis2 = pown(iz - zcenter, 2);

             z_ind = iz * slice;

             for (iy = yid; iy < shape.s1; iy += ystride){
                 yz_dis2 = pown(iy - ycenter, 2) + z_dis2;

                 yz_ind = z_ind + iy*shape.s2;

                 for (ix = xid; ix < shape.s2; ix += xstride){

                     xyz_dis2 = pown(ix - xcenter, 2) + yz_dis2;

                     if ((xyz_dis2 <= maxdis2) && (xyz_dis2 >= mindis2)){
                         xyz_ind = ix + yz_ind;
                         restspace[xyz_ind] += 1;
                     }
                 }
             }
         }
    }
}
