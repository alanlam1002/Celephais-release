// Mellin throat recursion, the stage (ii) port (code round 463), item 5: a minimal JSON READER -- the frozen decisions file
// (spinning_trumpet scripts/n2a_nl.py N2A_FREEZE; its predecessors' N2A_IDENT_PREV) is JSON with objects, arrays, strings, numbers, true /
// false / null; this parses exactly that (UTF-8 passed through, \u escapes of the BMP).  Objects keep their keys' order (Python's dicts).
#pragma once
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mk {
namespace js {

struct Val {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false;
    double n = 0.0;
    std::string s;
    std::vector<Val> a;
    std::vector<std::pair<std::string, Val>> o;
    const Val* get(const std::string& k) const {
        for (auto& kv : o)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
    bool has(const std::string& k) const { return get(k) != nullptr; }
};

class Parser {
  public:
    explicit Parser(const std::string& s) : s_(s) {}
    Val parse() {
        Val v = value();
        ws();
        if (i_ != s_.size()) throw std::runtime_error("mk::js: trailing characters");
        return v;
    }

  private:
    const std::string& s_;
    size_t i_ = 0;
    void ws() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\n' || s_[i_] == '\r' || s_[i_] == '\t')) i_++;
    }
    char peek() {
        ws();
        if (i_ >= s_.size()) throw std::runtime_error("mk::js: unexpected end");
        return s_[i_];
    }
    void expect(char c) {
        if (peek() != c) throw std::runtime_error(std::string("mk::js: expected '") + c + "'");
        i_++;
    }
    std::string str() {
        expect('"');
        std::string o;
        while (i_ < s_.size() && s_[i_] != '"') {
            char c = s_[i_++];
            if (c != '\\') { o += c; continue; }
            char e = s_[i_++];
            switch (e) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    unsigned cp = std::stoul(s_.substr(i_, 4), nullptr, 16);
                    i_ += 4;
                    if (cp < 0x80) o += char(cp);
                    else if (cp < 0x800) { o += char(0xC0 | (cp >> 6)); o += char(0x80 | (cp & 0x3F)); }
                    else { o += char(0xE0 | (cp >> 12)); o += char(0x80 | ((cp >> 6) & 0x3F)); o += char(0x80 | (cp & 0x3F)); }
                    break;
                }
                default: throw std::runtime_error("mk::js: bad escape");
            }
        }
        i_++;
        return o;
    }
    Val value() {
        char c = peek();
        Val v;
        if (c == '{') {
            i_++;
            v.t = Val::OBJ;
            if (peek() == '}') { i_++; return v; }
            while (true) {
                std::string k = str();
                expect(':');
                v.o.emplace_back(k, value());
                if (peek() == ',') { i_++; continue; }
                expect('}');
                return v;
            }
        }
        if (c == '[') {
            i_++;
            v.t = Val::ARR;
            if (peek() == ']') { i_++; return v; }
            while (true) {
                v.a.push_back(value());
                if (peek() == ',') { i_++; continue; }
                expect(']');
                return v;
            }
        }
        if (c == '"') { v.t = Val::STR; v.s = str(); return v; }
        if (s_.compare(i_, 4, "true") == 0) { i_ += 4; v.t = Val::BOOL; v.b = true; return v; }
        if (s_.compare(i_, 5, "false") == 0) { i_ += 5; v.t = Val::BOOL; return v; }
        if (s_.compare(i_, 4, "null") == 0) { i_ += 4; return v; }
        if (s_.compare(i_, 3, "NaN") == 0) { i_ += 3; v.t = Val::NUM; v.n = std::nan(""); return v; }
        if (s_.compare(i_, 8, "Infinity") == 0) { i_ += 8; v.t = Val::NUM; v.n = HUGE_VAL; return v; }
        if (s_.compare(i_, 9, "-Infinity") == 0) { i_ += 9; v.t = Val::NUM; v.n = -HUGE_VAL; return v; }
        const char* b = s_.c_str() + i_;
        char* e = nullptr;
        v.n = std::strtod(b, &e);
        if (e == b) throw std::runtime_error("mk::js: bad value");
        i_ += size_t(e - b);
        v.t = Val::NUM;
        return v;
    }
};

inline Val load(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("mk::js: cannot read " + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string s = ss.str();
    return Parser(s).parse();
}

}  // namespace js
}  // namespace mk
