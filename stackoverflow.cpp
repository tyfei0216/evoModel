#include<stdio.h>
#include<signal.h>
#include <unistd.h>

int aaa;
void sigint_handler (int sig ) 
{
    aaa += 1;
    printf("So you think you can stop the bomb with ctrl-c, do you?\n");
    // sleep(2);
    if (aaa>3)
        _exit(0);
}
int foo(int x) {
    int x1; 
    printf("address of x: %lld\n", (long long)&x1);
    printf("%d\n", x);
    // sleep(1);
    foo(x+1); 
    return 0;
}

int main() {
    aaa = 0;
    printf("address of foo: %lld\n", (long long)&foo);
    printf("address of main: %lld\n", (long long)&main);
    printf("address of aaa: %lld\n", (long long)&aaa);
    signal(SIGINT, sigint_handler);
    foo(0);
}