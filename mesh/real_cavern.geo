// Gmsh project created on Fri Feb  7 15:22:45 2020
//+
Point(1) = {-1000, -1000, 0, 1.0};
//+
Point(2) = {-1000, 1000, 0, 1.0};
//+
Point(3) = {1000, 1000, 0, 1.0};
//+
Point(4) = {1000, -1000, 0, 1.0};
//+
Point(5) = {-1000, 0, 0, 1.0};
//+
Point(6) = {-1000, 300, 0, 1.0};
//+
Point(7) = {-1000, -500, 0, 1.0};
//+
Point(8) = {-1000, -300, 0, 1.0};
//+
Point(9) = {-1000, 500, 0, 1.0};
//+
Point(10) = {-800, 300, 0, 1.0};
//+
Point(11) = {-800, -300, 0, 1.0};
//+
Line(1) = {7, 1};
//+
Line(2) = {1, 4};
//+
Line(3) = {4, 3};
//+
Line(4) = {3, 2};
//+
Line(5) = {2, 9};
