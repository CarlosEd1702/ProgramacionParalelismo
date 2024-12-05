#include <iostream>
#include <cuda_runtime.h>

struct Node {
    int data;
    Node* next;
};

// Puntero global para la lista en memoria unificada
__managed__ Node* head = nullptr;

// Insertar nodo al final de la lista
__host__ __device__ void insertAtEnd(int value) {
    Node* newNode = new Node;
    newNode->data = value;
    newNode->next = nullptr;

    if (head == nullptr) {
        head = newNode;
    }
    else {
        Node* current = head;
        while (current->next != nullptr) {
            current = current->next;
        }
        current->next = newNode;
    }
}

// Recorrer e imprimir la lista (solo desde CPU)
__host__ void traverseList() {
    Node* current = head;
    while (current != nullptr) {
        std::cout << current->data << " -> ";
        current = current->next;
    }
    std::cout << "NULL" << std::endl;
}

// Kernel para realizar operaciones en GPU
__global__ void gpuInsert() {
    insertAtEnd(50);  // Insertar nodo desde GPU
    insertAtEnd(60);
}

int main() {
    // Operaciones desde CPU
    insertAtEnd(10);
    insertAtEnd(20);
    insertAtEnd(30);

    std::cout << "Lista desde CPU antes de operaciones en GPU: ";
    traverseList();

    // Llamar al kernel para realizar inserciones en GPU
    gpuInsert << <1, 1 >> > ();
    cudaDeviceSynchronize();  // Esperar a que el GPU termine

    std::cout << "Lista desde CPU después de operaciones en GPU: ";
    traverseList();

    return 0;
}
