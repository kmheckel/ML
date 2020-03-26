#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

int main(int argc, char *argv[]){

  struct addrinfo *result;    		// to store results
  struct addrinfo *cur_result;    	// to store results

  struct addrinfo hints;      		// to indicate what information we want

  struct sockaddr_in *saddr;  		// to reference an IPv4 address
  struct sockaddr_in6 *saddr6;  	// to reference an IPv6 address

  int s; //for error checking00

	char* textaddr = calloc(INET6_ADDRSTRLEN, 1);

  //TODO: Complete the lab
  //
  // Outline:
  //   - set up the hints
	memset(&hints, 0, sizeof(struct addrinfo));
	hints.ai_protocol = IPPROTO_TCP;
  //   - perform the getaddrinfo()
	if( (s = getaddrinfo(argv[1], NULL, &hints, &result)) != 0)
	{
		fprintf(stderr, "getaddrinfo: %s\n",gai_strerror(s));
		exit(1);
	}
  //   - iterate over all the results, in linked-list fashion

	for (cur_result = result; cur_result != NULL; cur_result = cur_result->ai_next)
	{
  	//   - print each resolved ip address
		printf("%s has ", argv[1]);
		if (cur_result->ai_family == AF_INET6) 
		{
			saddr6 = (struct sockaddr_in6 *) cur_result->ai_addr;
			inet_ntop(cur_result->ai_family, &(saddr6->sin6_addr), textaddr, INET6_ADDRSTRLEN);
			printf("IPV6 ");
			printf("address %s\n", textaddr);
		}
		else
		{
			saddr = (struct sockaddr_in *) cur_result->ai_addr;
			printf("address %s\n", inet_ntoa(saddr->sin_addr));
		}
	}

  //   - cleanup the results with freaddrinfo(result)
  //free the addrinfo struct
  freeaddrinfo(result);
  
  return 0; //success
}
