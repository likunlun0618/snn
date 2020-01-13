#ifndef PARSE_H
#define PARSE_H

#include <string>
#include <vector>

namespace snn
{

namespace parse
{

using std::string;
using std::vector;

vector<string> readFile(string filename);
vector<string> deleteComments(vector<string> &lines);
vector<string> deleteEmptyLines(vector<string> &lines);
vector<string> split(string s, string separator);
vector<int> readArray(string s, string separator);
string parseItem(string s, string arg);

} // namespace Parse

} // namespace snn

#endif // PARSE_H
