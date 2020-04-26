// Gmsh project created on Tue Apr 21 20:29:07 2020
//+
Point(1) = {-1, -1, 0, 1.0};
//+
Point(2) = {-1, 1, 0, 1.0};
//+
Point(3) = {1, 1, 0, 1.0};
//+
Point(4) = {1, -1, 0, 1.0};
//+
Point(5) = {-1, 0.4, 0, 0.1};
//+
Point(6) = {-0.8, 0.2, 0, 0.1};
//+
Point(7) = {-1, 0.2, 0, 0.1};
//+
Point(8) = {-1, -0.2, 0, 0.1};
//+
Point(9) = {-0.8, -0.2, 0, 0.1};
//+
Point(10) = {-1, -0.4, 0, 0.1};
//+
Line(1) = {5, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
//+
Line(4) = {4, 1};
//+
Line(5) = {1, 10};
//+
Circle(6) = {5, 7, 6};
//+
Circle(7) = {9, 8, 10};
//+
Line(8) = {6, 9};
//+
Curve Loop(1) = {1, 2, 3, 4, 5, -7, -8, -6};
//+
Plane Surface(1) = {1};
//+
Physical Curve("cavern", 11) = {6, 8, 7};
//+
Physical Surface("domain", 1) = {1};
//+
Physical Curve("top", 21) = {2};
//+
Physical Curve("right", 22) = {3};
