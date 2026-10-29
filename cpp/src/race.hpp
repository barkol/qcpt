// Problem readers shared by the command line front end and the race scheduler.
#pragma once
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "beam.hpp"
#include "template_search.hpp"

namespace qcpt {

inline std::vector<std::string> read_names(std::istream& in) { int k; in >> k; std::vector<std::string> v(k); for (auto& s : v) in >> s; return v; }

// tokens: n m inputs[m] target[m] tp(0|1) [phase[m]]  max_cost beam lam depth div noise seed nthreads gpu  lib K names..  [allow M idx..]
inline void read_beam(std::istream& in, BeamParams& P, std::vector<std::string>& lib_names) {
    int m, flag, gpu; std::string tag; in >> P.n >> m; P.inputs.resize(m); P.target.resize(m);
    for (int& v : P.inputs) in >> v;
    for (int& v : P.target) in >> v;
    in >> flag; if (flag) { P.target_phase.resize(m); for (int& v : P.target_phase) in >> v; }
    in >> P.max_cost >> P.beam >> P.lam >> P.depth_side >> P.diversity >> P.noise >> P.seed >> P.nthreads >> gpu >> tag; P.gpu = gpu != 0; lib_names = read_names(in);
    in >> std::ws; if (in.peek() == 'a') { std::string a; in >> a >> m; P.allow.resize(m); for (int& v : P.allow) in >> v; }          // optional coupler filter
}

// tokens: n m inputs[m] target[m]  max_cost f1 fm f2 nproc seed  ga K names..  gb K names..  mid K names..
inline void read_template(std::istream& in, TemplateParams& P) {
    int m; std::string tag; in >> P.n >> m; P.inputs.resize(m); P.target.resize(m);
    for (int& v : P.inputs) in >> v;
    for (int& v : P.target) in >> v;
    in >> P.max_cost >> P.f1 >> P.fm >> P.f2 >> P.nproc >> P.seed >> tag; P.ga = read_names(in); in >> tag; P.gb = read_names(in); in >> tag; P.middles = read_names(in);
}

inline Library select_library(int n, const std::vector<std::string>& names, const std::vector<int>& allow = {}) {
    Library lib; int i = 0;
    for (const Gate& g : make_library(n, false)) { const bool ok = (names.empty() || std::find(names.begin(), names.end(), g.name) != names.end()) && (allow.empty() || std::binary_search(allow.begin(), allow.end(), i)); if (ok) lib.push_back(g); ++i; }
    return lib;
}

// Race of all members on one thread pool. stdin: start best round_limit_s nmembers, then policy numbers alpha gamma (start delay factor, weight multiplier of a round winner), then per member "member NAME beam|template OFFSET" and the member's problem.
// Output: one STEP line per improvement, then RESULT and the gates of the last winner.
int run_race(std::istream& in);

}  // namespace qcpt
