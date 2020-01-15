// Gmsh project created on Wed Jan 15 12:36:44 2020
//+
Point(1) = {-1, 1, 0, 1.0};
//+
Point(2) = {-1, -1, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {1, -1, 0, 1.0};
//+
Line(1) = {1, 3};
//+
Line(2) = {3, 4};
//+
Line(3) = {4, 2};
//+
Line(4) = {2, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Surface(1) = {1};
