#pragma once

#include <string>
#include <vector>

typedef std::vector<std::string> string_vector;

auto read_file_to_string_vector( const std::string & filename,
				 string_vector & file_text ) -> bool;

