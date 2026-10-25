// Build the reciprocal, language-labelled largest component; no solver decisions enter filtering.
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <zlib.h>
namespace fs = std::filesystem;
using U = uint32_t;
using K = uint64_t;
void require(bool b, const std::string &s) { if (!b) throw std::runtime_error(s); }
std::string quote(const std::string &s) { std::string o="\""; for(char c:s) { if(c=='"'||c=='\\')o+='\\'; require(static_cast<unsigned char>(c)>=32,"Invalid JSON string"); o+=c; } return o+'"'; }
template<class T> void npy(const fs::path &p, const std::vector<T> &a, std::string dtype, size_t cols=0) {
    std::ofstream f(p,std::ios::binary); require(bool(f),"Cannot write "+p.string());
    std::string h="{'descr': '"+dtype+"', 'fortran_order': False, 'shape': ("+std::to_string(cols?a.size()/cols:a.size())+","+(cols?" "+std::to_string(cols)+",":"")+"), }";
    h+=std::string((64-(10+h.size()+1)%64)%64,' '); h+='\n';
    f.write("\x93NUMPY\x01\x00",8); uint16_t len=h.size(); f.put(len&255); f.put(len>>8); f<<h;
    f.write(reinterpret_cast<const char*>(a.data()),a.size()*sizeof(T)); require(bool(f),"NPY write failed");
}
std::vector<U> read_cache(const fs::path &p) {
    std::ifstream f(p,std::ios::binary); char prefix[10]; f.read(prefix,10); require(f && std::string(prefix,8)==std::string("\x93NUMPY\x01\x00",8),"Invalid cache");
    size_t h=(unsigned char)prefix[8]+256*(unsigned char)prefix[9]; f.seekg(h,std::ios::cur); auto pos=f.tellg(); f.seekg(0,std::ios::end); auto bytes=f.tellg()-pos; require(bytes>=0 && bytes%8==0,"Invalid cache length"); f.seekg(pos); std::vector<U>a(size_t(bytes)/4); f.read(reinterpret_cast<char*>(a.data()),bytes); require(bool(f),"Cache read failed"); return a;
}
struct DSU { std::vector<U> p,sz; explicit DSU(size_t n):p(n),sz(n,1){std::iota(p.begin(),p.end(),0);} U root(U x){while(p[x]!=x){p[x]=p[p[x]];x=p[x];}return x;} void join(U x,U y){x=root(x);y=root(y);if(x==y)return;if(sz[x]<sz[y])std::swap(x,y);p[y]=x;sz[x]+=sz[y];} };
int main(int argc,char**argv) { try {
    std::map<std::string,std::string> args; for(int i=1;i<argc;i+=2){require(i+1<argc,"Expected --key value");args[argv[i]]=argv[i+1];}
    for(auto &[k,v]:args)require(k=="--followers"||k=="--languages"||k=="--output"||k=="--cache-dir"||k=="--top-k","Unknown argument "+k);
    require(args.count("--followers") && args.count("--cache-dir"),"Usage: build_graph --followers FILE.gz --cache-dir DIR [--languages TSV --output DIR --top-k 20]");
    fs::path source=args["--followers"], cache=args["--cache-dir"]; fs::create_directories(cache);
    const std::string fingerprint=fs::absolute(source).string()+"\n"+std::to_string(fs::file_size(source))+"\n"+std::to_string(fs::last_write_time(source).time_since_epoch().count());
    std::vector<U> pairs; fs::path marker=cache/"source_fingerprint.txt";
    if(fs::exists(marker)) { std::ifstream f(marker);std::string old((std::istreambuf_iterator<char>(f)),{});require(old==fingerprint,"Cache source changed; use a fresh cache directory");pairs=read_cache(cache/"reciprocal_edges.npy"); }
    else {
        gzFile gz=gzopen(source.c_str(),"rb");require(gz,"Cannot open gzip");gzbuffer(gz,1<<20);
        std::vector<K> forward,back; K rows=0,self=0,hash=14695981039346656037ULL; char line[256];
        while(gzgets(gz,line,sizeof(line))) { std::string s(line); require(s.back()=='\n'||gzeof(gz),"Overlong row"); for(unsigned char c:s){hash^=c;hash*=1099511628211ULL;}
            if(!s.empty()&&s.back()=='\n')s.pop_back();if(!s.empty()&&s.back()=='\r')s.pop_back();auto comma=s.find(',');require(comma!=s.npos && s.find(',',comma+1)==s.npos,"Expected headerless u,v rows");size_t a,b;K u=std::stoull(s.substr(0,comma),&a),v=std::stoull(s.substr(comma+1),&b);require(a==comma&&b==s.size()-comma-1&&u<=UINT32_MAX&&v<=UINT32_MAX,"Invalid user ID");++rows;if(u==v){++self;continue;} (u<v?forward:back).push_back((std::min(u,v)<<32)|std::max(u,v));
        }
        int err;gzerror(gz,&err);require(err==Z_OK||err==Z_STREAM_END,"Gzip read error");require(gzclose(gz)==Z_OK,"Gzip close error");
        auto unique=[](std::vector<K>&v){std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());}; unique(forward);unique(back);
        std::vector<K> both;std::set_intersection(forward.begin(),forward.end(),back.begin(),back.end(),std::back_inserter(both));
        std::vector<U> observed;observed.reserve(2*(forward.size()+back.size()));for(auto *v:{&forward,&back})for(K e:*v){observed.push_back(e>>32);observed.push_back(U(e));}std::sort(observed.begin(),observed.end());observed.erase(std::unique(observed.begin(),observed.end()),observed.end());
        pairs.reserve(2*both.size());for(K e:both){pairs.push_back(e>>32);pairs.push_back(U(e));}auto reciprocal_vertices=pairs;std::sort(reciprocal_vertices.begin(),reciprocal_vertices.end());reciprocal_vertices.erase(std::unique(reciprocal_vertices.begin(),reciprocal_vertices.end()),reciprocal_vertices.end());npy(cache/"reciprocal_edges.npy",pairs,"<u4",2);
        std::ofstream st(cache/"raw_graph_stats.json");st<<"{\n\"raw_rows\":"<<rows<<",\n\"self_loop_rows\":"<<self<<",\n\"directed_unique_nonself_edges\":"<<forward.size()+back.size()<<",\n\"duplicate_nonself_rows\":"<<rows-self-forward.size()-back.size()<<",\n\"observed_nonself_vertices\":"<<observed.size()<<",\n\"reciprocal_undirected_edges\":"<<both.size()<<",\n\"reciprocal_vertices\":"<<reciprocal_vertices.size()<<",\n\"fraction_directed_nonself_edges_reciprocated\":"<<(forward.size()+back.size()?2.0*both.size()/(forward.size()+back.size()):0)<<",\n\"decoded_input_fnv1a64\":"<<quote(std::to_string(hash))<<"\n}\n";st.close();require(bool(st),"Stats write failed");std::ofstream(marker)<<fingerprint;
    }
    std::cerr<<"Reciprocal edges: "<<pairs.size()/2<<'\n'; if(!args.count("--languages"))return 0;
    require(args.count("--output"),"Need --output with --languages");int top=args.count("--top-k")?std::stoi(args["--top-k"]):20;require(top>0&&top<=65535,"Invalid top-k");
    std::ifstream lang(args["--languages"]);require(bool(lang),"Cannot open languages");std::string line;std::getline(lang,line);require(line.rfind("user_id\tlanguage",0)==0,"Invalid languages header");std::map<U,std::string> users;std::map<std::string,K> counts;
    while(std::getline(lang,line)){std::istringstream s(line);std::string id,l;std::getline(s,id,'\t');std::getline(s,l,'\t');K v=std::stoull(id);require(v<=UINT32_MAX&&!l.empty(),"Invalid language record");require(users.emplace(U(v),l).second,"Duplicate user language");++counts[l];}
    std::vector<std::string> chosen;for(auto &[l,c]:counts)chosen.push_back(l);std::sort(chosen.begin(),chosen.end(),[&](auto &a,auto &b){return counts[a]!=counts[b]?counts[a]>counts[b]:a<b;});if(chosen.size()>size_t(top))chosen.resize(top);std::map<std::string,U> selected;for(auto &l:chosen)selected[l]=0;
    std::vector<U> ids; std::unordered_map<U,U> index; for(auto &[id,l]:users)if(selected.count(l)){index[id]=ids.size();ids.push_back(id);}require(!ids.empty(),"No eligible users");DSU d(ids.size());std::vector<U> filtered;
    K labeled_edges=0;for(size_t i=0;i<pairs.size();i+=2){U u=pairs[i],v=pairs[i+1];if(users.count(u)&&users.count(v))++labeled_edges;auto a=index.find(u),b=index.find(v);if(a!=index.end()&&b!=index.end()){d.join(a->second,b->second);filtered.push_back(a->second);filtered.push_back(b->second);}}
    U best=0;for(U i=0;i<ids.size();++i)if(d.sz[d.root(i)]>d.sz[d.root(best)])best=i;U root=d.root(best);std::vector<U> out_ids,remap(ids.size(),UINT32_MAX);std::map<std::string,K> population;
    for(U i=0;i<ids.size();++i)if(d.root(i)==root){remap[i]=out_ids.size();out_ids.push_back(ids[i]);++population[users.at(ids[i])];}
    std::map<std::string,uint16_t> group;for(auto &[l,c]:population)group[l]=group.size();std::vector<uint16_t> labels;for(U id:out_ids)labels.push_back(group.at(users.at(id)));std::vector<U> edges;for(size_t i=0;i<filtered.size();i+=2)if(remap[filtered[i]]!=UINT32_MAX){edges.push_back(remap[filtered[i]]);edges.push_back(remap[filtered[i+1]]);}
    fs::path out=args["--output"];fs::create_directories(out);require(!fs::exists(out/"edges.npy"),"Output already contains edges.npy");npy(out/"edges.npy",edges,"<u4",2);npy(out/"labels.npy",labels,"<u2");npy(out/"user_ids.npy",out_ids,"<u4");fs::copy_file(cache/"raw_graph_stats.json",out/"raw_graph_stats.json",fs::copy_options::overwrite_existing);
    std::ofstream meta(out/"metadata.json");meta<<"{\"top_k_requested\":"<<top<<",\"eligible_labeled_users\":"<<users.size()<<",\"top_language_users_before_lcc\":"<<ids.size()<<",\"reciprocal_edges_both_labeled\":"<<labeled_edges<<",\"reciprocal_edges_top_languages_before_lcc\":"<<filtered.size()/2<<",\"n\":"<<out_ids.size()<<",\"m\":"<<edges.size()/2<<",\"selection_policy\":\"top languages by all eligible labelled users, then induced reciprocal graph, then largest component; ties lexical language or smallest original user ID\",\"groups\":[";
    bool first=true;for(auto &[l,g]:group){if(!first)meta<<',';first=false;meta<<"{\"id\":"<<g<<",\"language\":"<<quote(l)<<",\"population\":"<<population[l]<<"}";}meta<<"],\"selected_languages_before_lcc\":[";first=true;for(auto &l:chosen){if(!first)meta<<',';first=false;meta<<"{\"language\":"<<quote(l)<<",\"eligible_users\":"<<counts[l]<<"}";}meta<<"],\"dropped_languages_in_lcc\":[";first=true;for(auto &l:chosen)if(!group.count(l)){if(!first)meta<<',';first=false;meta<<quote(l);}meta<<"]}\n";require(bool(meta),"Metadata write failed");std::cerr<<"Final graph: "<<out_ids.size()<<" users, "<<edges.size()/2<<" edges, "<<group.size()<<" languages\n";
} catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;} }
