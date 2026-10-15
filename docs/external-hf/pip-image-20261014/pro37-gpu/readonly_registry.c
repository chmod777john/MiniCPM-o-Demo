#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sendfile.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
static int listener=-1;
static const char *root="/registry/blobs/sha256";
static const char *names[]={"worker","gateway"};
static const char *manifests[]={"c35318d4cc66ffc9d15a6d21cf69f8c5d46526ebd7ff1b1144f8a2604aa3f8f6","143bdde56a917513253998c8ccfa032fc84ffcb72e7c38b1da95bbbcd7a5e3cd"};
static const char *blobs[]={"1c59224885e4cd11143c8280cb86d39184fe1d63bfe5a59a7e44bf072edede94","2720663bd48d29c34436b222dd78358094e9945541a00b08b9b8a5bce498e79c","48d6767f7b03f8beee76e2fcfebaa48999eb7372eed7c05d0f51c4025c7532bf","4f4fb700ef54461cfa02571ae0db9a0dc1e0cdb5577484a6d75e68dc38e8acc1","5dc92ad27f736e3cdae5da253417d61c0ba5b76493da099af0532f39d7ded43b","66a73b1dcbb80ac11d59cdbafbe61dd7648d1b322d945e0c624cd855cddb4237","7fd1ebe418b5079d4aa36f416af9b143d127bc75b0155d71cb5fd559b5e04d39","8019b7922ecc4170e66956022c64c210a25ca4cfacd04fb0a022b470195bb9f2","84771b95dccbc7379bfcdf79204272c472e7f523ea4ee7af43783bd6328b016c","8acfaf9989c845277276331ea14f6c5ad0d2c54964824de67474a8583e29c1e5","a010d0edffc90aac95cc9ea82c8ffafe5238e6b84862c6a5ebdf9f8fb91ebdf3","b094e9eea2ad246129947c9f940df7881328c2ee5a47305d87742d6453ff7c7b","b1912a38d0c5acd20178b249da47521bbbc4f2869c0920cc18578791c5144d29","b22eadbaca205d3fc724a4a6b9ebfd7e83539e23ba36b325fdcdf37719a2f047","bbf3b65e65c07e6a1cf53c8ff7ee2feae1c6629f1333f12c6802d03ed8980794","c21726021fad08adbd9088c97fde214bbbbaaf36dae7270c70a63e1e02bd6e4b","c85d294956374c22c738b1e724a606a14ef5a943130b8ce9b6e7fbba2da31875","f4d90713de05e34d9261fd5f119037ea4a10b3270b25492e62b4f967648e27a3"};
static int writeall(int fd,const void *buf,size_t len){const char *p=buf;while(len){ssize_t n=send(fd,p,len,MSG_NOSIGNAL);if(n<=0)return -1;p+=n;len-=n;}return 0;}
static void simple(int fd,int status,const char *reason,int head){char h[512];int n=snprintf(h,sizeof h,"HTTP/1.1 %d %s\r\nDocker-Distribution-Api-Version: registry/2.0\r\nContent-Type: application/json\r\nContent-Length: 2\r\nConnection: close\r\n\r\n",status,reason);writeall(fd,h,n);if(!head)writeall(fd,"{}",2);}
static int known(const char *sha){for(size_t i=0;i<sizeof(blobs)/sizeof(*blobs);i++)if(!strcmp(sha,blobs[i]))return 1;return 0;}
static void handle(int fd){char req[16384];size_t used=0;struct timeval tv={.tv_sec=15};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&tv,sizeof tv);tv.tv_sec=120;setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&tv,sizeof tv);
while(used<sizeof(req)-1){ssize_t n=recv(fd,req+used,sizeof(req)-1-used,0);if(n<=0)return;used+=n;req[used]=0;if(strstr(req,"\r\n\r\n"))break;}
char method[16],path[1024],ver[16];if(sscanf(req,"%15s %1023s %15s",method,path,ver)!=3){simple(fd,400,"Bad Request",0);return;}
int head=!strcmp(method,"HEAD");if(strcmp(method,"GET")&&!head){simple(fd,405,"Method Not Allowed",0);return;}
fprintf(stdout,"%s %s\n",method,path);fflush(stdout);
if(!strcmp(path,"/v2/")||!strcmp(path,"/v2")){simple(fd,200,"OK",head);return;}
char role[32],kind[32],sha[128],tail[4];int fields=sscanf(path,"/v2/%31[^/]/%31[^/]/sha256:%127[0-9a-f]%3s",role,kind,sha,tail);
if(fields!=3||strlen(sha)!=64){simple(fd,404,"Not Found",head);return;}
int rid=-1;for(int i=0;i<2;i++)if(!strcmp(role,names[i]))rid=i;
int manifest=!strcmp(kind,"manifests");if(rid<0||(!manifest&&strcmp(kind,"blobs"))||(manifest?strcmp(sha,manifests[rid]):!known(sha))){simple(fd,404,"Not Found",head);return;}
char file[256];snprintf(file,sizeof file,"%s/%s",root,sha);int f=open(file,O_RDONLY|O_NOFOLLOW);struct stat st;if(f<0||fstat(f,&st)||!S_ISREG(st.st_mode)){if(f>=0)close(f);simple(fd,404,"Not Found",head);return;}
char h[768];int n=snprintf(h,sizeof h,"HTTP/1.1 200 OK\r\nDocker-Distribution-Api-Version: registry/2.0\r\nDocker-Content-Digest: sha256:%s\r\nContent-Type: %s\r\nContent-Length: %lld\r\nConnection: close\r\n\r\n",sha,manifest?"application/vnd.oci.image.manifest.v1+json":"application/octet-stream",(long long)st.st_size);
if(!writeall(fd,h,n)&&!head){off_t off=0;while(off<st.st_size){ssize_t r=sendfile(fd,f,&off,64*1024*1024);if(r<0&&errno==EINTR)continue;if(r<=0)break;}}
close(f);}
static void stop(int sig){(void)sig;if(listener>=0)close(listener);_exit(0);}
int main(int argc,char **argv){int port=argc>1?atoi(argv[1]):18537;if(port<1024||port>65535)return 2;signal(SIGPIPE,SIG_IGN);signal(SIGCHLD,SIG_IGN);signal(SIGTERM,stop);signal(SIGINT,stop);listener=socket(AF_INET,SOCK_STREAM,0);if(listener<0){perror("socket");return 1;}struct sockaddr_in a={.sin_family=AF_INET,.sin_port=htons(port)};inet_pton(AF_INET,"127.0.0.1",&a.sin_addr);if(bind(listener,(void*)&a,sizeof a)||listen(listener,32)){perror("bind/listen");return 1;}printf("READ_ONLY registry 127.0.0.1:%d worker=%s gateway=%s\n",port,manifests[0],manifests[1]);fflush(stdout);for(;;){int fd=accept(listener,NULL,NULL);if(fd<0){if(errno==EINTR)continue;perror("accept");return 1;}pid_t child=fork();if(child==0){close(listener);listener=-1;handle(fd);close(fd);_exit(0);}close(fd);}}
