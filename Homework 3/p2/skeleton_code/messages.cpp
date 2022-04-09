#include <cassert>
#include <iostream>
#include <mpi.h>
#include <vector>
#include <list>
#include <queue>
#include <chrono>
#include <thread>

//
// Data Structures
//

struct Message {
    double x1, x2;
    int z1, z2;
};

template<template <typename, typename> class Container>
struct MessageArray {
    Container<double, std::allocator<double>> x1, x2;
    Container<int, std::allocator<int>> z1, z2;
};


//
// Utility functions (not important)
//

void print(Message m) {
    std::cout << m.x1  << " " << m.x2 << " " << m.z1  << " " << m.z2 << std::endl;
}

template<template <typename, typename> class Container>
void print(const MessageArray<Container>& messages) {
    for(const auto& x :  messages.x1)
        std::cout << x << " ";
    std::cout << std::endl;
    for(const auto& x :  messages.x2)
        std::cout << x << " ";
    std::cout << std::endl;
    for(const auto& x :  messages.z1)
        std::cout << x << " ";
    std::cout << std::endl;
    for(const auto& x :  messages.z2)
        std::cout << x << " ";
    std::cout << std::endl;
  }

template<typename Container>
void print(const Container& messages) {
  for(const Message& m :  messages)
      print(m);
}

template<typename Container>
void init_messages(Container& messages, int n) {
  for(int i = 0; i < n; ++i)
      messages.push_back({
        4.0 * i, 4.0 * i + 1, 4 * i + 2, 4 * i + 3,
      });
}

template<typename Container>
bool test_aos_type(const Container& messages) {
  int i = 0;
  for(const Message& m :  messages) {
      if(m.x1 != 4 * i || m.x2 != 4 * i + 1 || m.z1 != 4 * i + 2 || m.z2 != 4 * i + 3) {
          std::cout << "Error - Test failed!" << std::endl;
          return false;
      }
      ++i;
  }
  std::cout << "Solution apears to be correct!" << std::endl;
  return true;
}

template<template <typename, typename> class Container>
bool test_soa_type(const MessageArray<Container>& messages) {
    
    int i = 0;
    for(const auto& x :  messages.x1)
        if(x != 4 * i++) {
            std::cout << "Error - Test failed!" << std::endl;
            return false;
        }
    
    i = 0;
    for(const auto& x :  messages.x2)
        if(x != 4 * i++ + 1) {
            std::cout << "Error - Test failed!" << std::endl;
            return false;
        }

    i = 0;
    for(const auto& x :  messages.z1)
        if(x != 4 * i++ + 2) {
            std::cout << "Error - Test failed!" << std::endl;
            return false;
        }
    
    i = 0;
    for(const auto& x :  messages.z2)
        if(x != 4 * i++ + 3) {
            std::cout << "Error - Test failed!" << std::endl;
            return false;
        }
        
    std::cout << "Solution apears to be correct!" << std::endl;
    return true;

  }


//
// Functions to complete
//

////////////////  Question a)  ////////////////
void init_base_type(MPI_Datatype& base_type) {
    
    /*
        TO DO: Initialize base_type to be a custom MPI_Datatype for sending a struct of type Message
    */
    MPI_Aint offsets[4];
    offsets[0] = offsetof(Message, x1);
    offsets[1] = offsetof(Message, x2);
    offsets[2] = offsetof(Message, z1);
    offsets[3] = offsetof(Message, z2);

    int blocklens[] = {1, 1, 1, 1};
    MPI_Datatype types[] = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT};
    
    MPI_Type_create_struct(4, blocklens, offsets, types, &base_type);
    MPI_Type_commit(&base_type);

}

////////////////  Question b)  ////////////////
template<typename Container>
void init_aos_type(const Container& c, const MPI_Datatype& base_type, MPI_Datatype& container_type) {

    /*
        TO DO: Initialize container_type to be a custom MPI_Datatype for sending a container of structs of type Message
    */

    const int n = c.size();
    std::vector<MPI_Datatype> types(n, base_type);
    std::vector<int> blocklens(n, 1);
    std::vector<MPI_Aint> offsets;
    MPI_Aint address;

    for (const auto &x : c) {
        MPI_Get_address(&x, &address);
        offsets.push_back(address);
    }

    MPI_Type_create_struct(n, blocklens.data(), offsets.data(), types.data(), &container_type);
    MPI_Type_commit(&container_type);  

}

////////////////  Question c)  ////////////////
template<typename T>
void init_aos_type(const std::vector<T>& c, const MPI_Datatype& base_type, MPI_Datatype& container_type) {

    /*
        TO DO: Initialize container_type to be a custom MPI_Datatype for sending a container of structs of type Message
    */

    // This allows the code to work before this stage is completed - please delete this line when solving question c
    // init_aos_type<bool>(false, base_type, container_type);
    const int n = c.size();
    MPI_Type_vector(n, 1, 1, base_type, &container_type);
    MPI_Type_commit(&container_type);

}

////////////////  Question d)  ////////////////
template<typename Container>
void init_soa_type(const Container& messages, MPI_Datatype& container_type) {

    const int n = messages.size();
    std::vector<MPI_Datatype> types = {MPI_DOUBLE, MPI_DOUBLE, MPI_INT, MPI_INT};
    std::vector<int> blocklens(n, 4)
    std::vector<MPI_Aint> offsets[4];
    
    std::vector<int> x1, x2;
    std::vector<double> z1, z2;

    for (const auto &x : messages) {
        x1.push_back(x.x1);
        x2.push_back(x.x2);
        z1.push_back(x.z1);
        z2.push_back(x.z2);
    }

    MPI_Type_create_struct(n, blocklens.data(), offsets.data(), types.data(), &container_type);
    MPI_Type_commit(&container_type); 

}

//
// The MPI Program
//

int main(int argc, char **argv)
{
    ////////////////   Init MPI    ////////////////

    unsigned int n = 10;
    MPI_Init(&argc, &argv);
    
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if (comm_size != 2) {
        std::cerr << "ERROR: Needs to run with 2 MPI processes." << std::endl;
        MPI_Finalize();
        return -1;
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ////////////////  Question a)  ////////////////
    if (rank == 0)
        std::cout << std::endl
                  << std::endl
                  << "Beginning of Question a)" << std::endl;

    
    ////////////////  Complete MPI Datatype for Message  ////////////////
    MPI_Datatype base_type;
    init_base_type(base_type);

    if(rank == 0) {
        std::vector<Message> messages;
        init_messages(messages, 1);
        MPI_Send(messages.data(), 1, base_type, 1, 42, MPI_COMM_WORLD);
    }

    if(rank == 1) {
        std::vector<Message> messages(1);
        MPI_Recv(messages.data(), 1, base_type, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print(messages);
        test_aos_type(messages);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    MPI_Barrier(MPI_COMM_WORLD);


    ////////////////  Question b)  ////////////////
    if (rank == 0)
        std::cout << std::endl
                  << std::endl
                  << "Beginning of Question b)" << std::endl;
    
    if(rank == 0) {
        std::deque<Message> messages;
        init_messages(messages, n);
        MPI_Datatype container_type;
        init_aos_type(messages, base_type, container_type);
        MPI_Send(MPI_BOTTOM, 1, container_type, 1, 42, MPI_COMM_WORLD);
        MPI_Type_free(&container_type);
    }

    if(rank == 1) {
        std::list<Message> messages(n);
        MPI_Datatype container_type;
        init_aos_type(messages, base_type, container_type);
        MPI_Recv(MPI_BOTTOM, 1, container_type, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print(messages);
        test_aos_type(messages);
        MPI_Type_free(&container_type);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    MPI_Barrier(MPI_COMM_WORLD);


    ////////////////  Question c)  ////////////////
    if (rank == 0)
        std::cout << std::endl
                  << std::endl
                  << "Beginning of Question c)" << std::endl;
    
    if(rank == 0) {
        std::vector<Message> messages;
        init_messages(messages, n);
        MPI_Datatype container_type;
        init_aos_type(messages, base_type, container_type);
        MPI_Send(messages.data(), 1, container_type, 1, 42, MPI_COMM_WORLD);
        MPI_Type_free(&container_type);
    }

    if(rank == 1) {
        std::list<Message> messages(n);
        MPI_Datatype container_type;
        init_aos_type(messages, base_type, container_type);
        MPI_Recv(MPI_BOTTOM, 1, container_type, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print(messages);
        test_aos_type(messages);
        MPI_Type_free(&container_type);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    MPI_Barrier(MPI_COMM_WORLD);


    ////////////////  Question d)  ////////////////
    if (rank == 0)
        std::cout << std::endl
                  << std::endl
                  << "Beginning of Question d)" << std::endl;

    if(rank == 0) {
        std::vector<Message> messages;
        init_messages(messages, n);
        MPI_Datatype container_type;
        init_aos_type(messages, base_type, container_type);
        MPI_Send(messages.data(), 1, container_type, 1, 42, MPI_COMM_WORLD);
        MPI_Type_free(&container_type);
    }

    if(rank == 1) {
        MessageArray<std::vector> messages;
        MPI_Datatype container_type;
        messages.x1.resize(n), messages.x2.resize(n);
        messages.z1.resize(n), messages.z2.resize(n);
        init_soa_type(messages, container_type);
        MPI_Recv(MPI_BOTTOM, 1, container_type, 0, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        print(messages);
        test_soa_type(messages);
        MPI_Type_free(&container_type);
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    MPI_Barrier(MPI_COMM_WORLD);

    
    MPI_Type_free(&base_type);
    MPI_Finalize();
    return 0;
}
