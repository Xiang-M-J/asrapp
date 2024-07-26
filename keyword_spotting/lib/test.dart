import 'dart:math';

norm(List<double> e) {

  double s = e.map((a) => a * a).reduce((a, b) => a  + b);
  return sqrt(s);
}

void main(){
  print(norm([1,2,3,4,5]));
}
