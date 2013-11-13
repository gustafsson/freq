path = "sonicawe"

$command_line_width = 80
bindir="sandbox/Contents/MacOS"
system("cp #{path} #{bindir}/#{path}")
Dir.chdir(bindir)
libraries = `ls ../Frameworks`.split
    
    # Setting install names
    #puts " Fixing install names ".center($command_line_width, "=")
    #puts "in binary: #{File.basename(path)}"
    libraries.each do |library|
        libname = "#{File.basename(library, ".*")}"
        
        libpath = "../Frameworks"
        newlibpath = "@executable_path/../Frameworks"
        libfile = "#{libpath}/#{library}"
        libtargetid = `otool -DX #{libfile}`.strip
        targetid = `otool -L #{path} | grep "#{libname}\\\\." | sed s/\\(.*\\)//`.strip
        newtargetid = "#{newlibpath}/#{library}"

    if (libtargetid != newtargetid)
        puts "updating library #{libtargetid} -> #{newtargetid}"
        unless system("install_name_tool -id #{newtargetid} #{libfile}")
            puts "Error: Could not set id #{newtargetid} in binary #{libfile}"
            exit(1)
        end
    end

        binary_uses_this_lib = !`otool -L #{path} | grep "#{targetid}"`.empty?
        next unless binary_uses_this_lib && !targetid.empty? && targetid!=newtargetid
        
        puts "Updating install name in #{path} for library #{library}"
        
        unless system("install_name_tool -change #{targetid} #{newtargetid} #{path}")
            puts "Error: Could not change install name for #{libpath}/#{libname} from #{targetid} to #{newtargetid} in binary #{path}"
            exit(1)
        end
    end
