//
// Created by max on 15.09.21.
//

#ifndef SDFGEN_DEFORM_H
#define SDFGEN_DEFORM_H

class Mapping{
public:
    static void mapPoint(dPoint &point){
        double z = point[2];
        // normalize and invert
        z = 1.0 - ((z + 1.0) / 2.0);
        // scale a new and invert back
        z = 1.0 - sqrt(z);
        // denormalize back to -1 to 1
        point[2] = z * 2.0 - 1.0;
    };

    static void mapPointBack(dPoint &point){
        double z = point[2];
        // normalize and invert
        z = 1.0 - ((z + 1.0) / 2.0);
        // scale a new and invert back
        z = 1.0 - (z * z);
        // denormalize back to -1 to 1
        point[2] = z * 2.0 - 1.0;
    };
};
#endif //SDFGEN_DEFORM_H
