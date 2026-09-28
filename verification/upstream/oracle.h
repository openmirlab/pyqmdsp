// JSON output for standalone upstream calls; never imports binding code.
// Reads: standard library. Values round-trip through 17 significant digits.
#pragma once
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Json {
    std::ostringstream body;
    bool first = true;
    static void value(std::ostream &s, const Json &v) { s << '{' << v.body.str() << '}'; }
    static void value(std::ostream &s, const std::string &v) { s << std::quoted(v); }
    static void value(std::ostream &s, const char *v) { value(s, std::string(v)); }
    static void value(std::ostream &s, bool v) { s << (v ? "true" : "false"); }
    template<class T> static void value(std::ostream &s, T v) {
        if (!std::isfinite(double(v))) throw std::runtime_error("non-finite reference output");
        s << std::setprecision(17) << v;
    }
    template<class T> static void value(std::ostream &s, const std::vector<T> &v) {
        s << '[';
        for (size_t i=0; i<v.size(); ++i) { if(i) s << ','; value(s,v[i]); }
        s << ']';
    }
    template<class T> void set(const std::string &key, const T &v) {
        if(!first) body << ',';
        first=false;
        body << std::quoted(key) << ':';
        value(body,v);
    }
    void save(const char *path) const {
        std::ofstream out(path);
        if(!out) throw std::runtime_error("cannot write reference file");
        value(out,*this); out << '\n';
    }
};
inline std::vector<double> signal(size_t n, double phase=0) {
    std::vector<double> x(n);
    for(size_t i=0;i<n;++i) x[i]=0.3*std::sin(0.071*i+phase)+0.17*std::cos(0.193*i)+double(int(i%17)-8)/100;
    return x;
}
