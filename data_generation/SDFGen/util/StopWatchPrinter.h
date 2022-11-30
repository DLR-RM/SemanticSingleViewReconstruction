//
// Created by max on 12.05.20.
//

#ifndef SDFGEN_STOPWATCHPRINTER_H
#define SDFGEN_STOPWATCHPRINTER_H


#include <string>
#include <iostream>
#include "StopWatch.h"

class StopWatchPrinter {
public:

    StopWatchPrinter(std::string context): m_context(context){
        m_sw.startTime();
        std::cout << "Start: " << m_context << std::endl;
    };

    void finish(){
        std::cout << "Done: " << m_context << ", in: " << m_sw.elapsed_time() << "s" << std::endl;
    }

private:
    std::string m_context;
    bool m_printed;
    StopWatch m_sw;
};


#endif //SDFGEN_STOPWATCHPRINTER_H
