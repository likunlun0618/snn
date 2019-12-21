#include <cstring>
#include <fstream>
#include <sstream>
#include "parse.h"

namespace parse
{

vector<string> readFile(string filename)
{
    std::ifstream file(filename);
    char buffer[1024];
    vector<string> lines;
    while (file.getline(buffer, sizeof(buffer)))
        lines.push_back(string(buffer));
    file.close();
    return lines;
}

vector<string> deleteComments(vector<string> &lines)
{
    vector<string> ret;
    for (string line : lines)
    {
        for (int i = 0; i < (int)line.length() - 1; ++i)
            if (line[i] == '/' && line[i + 1] == '/')
            {
                line = line.substr(0, i);
                break;
            }    
        ret.push_back(line);
    }
    return ret;
}

vector<string> deleteEmptyLines(vector<string> &lines)
{
    vector<string> ret;
    for (string &line : lines)
        if (line.length() > 0)
            ret.push_back(line);
    return ret;
}

vector<string> split(string s, string separator)
{
    char *token = strtok((char *)s.data(), (char *)separator.data());
    vector<string> ret;
    while (token != NULL)
    {
        ret.push_back(string(token));
        token = strtok(NULL, (char *)separator.data());
    }
    return ret;
}

vector<int> readArray(string s, string separator)
{
    vector<string> items = split(s, separator);
    vector<int> ret(items.size());
    for (int i = 0; i < items.size(); ++i)
        std::stringstream(items[i]) >> ret[i];
    return ret;
}

string parseItem(string item, string arg)
{
    string ret;
    if (item.length() < arg.length() + 2)
        return ret;
    ret = item.substr(arg.length() + 1, item.length() - arg.length() - 2);
    return ret;
}

} // namespace parse
