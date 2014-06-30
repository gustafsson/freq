####
# http://doc.qt.nokia.com/4.7-snapshot/deployment-mac.html
####

$cuda_library_path = "/usr/local/cuda/lib"
$custom_library_path = "/opt/local/lib"
$custom_library_path = "/usr/local/lib"
$compiler_library_path = "/opt/local/lib/gcc49"
$compiler_library_path = `xcode-select -p`[0..-2] + "/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk/usr/lib"
$command_line_width = 80

# Configuration
$custom_exec = "../src/sonicawe"
$custom_exec = ARGV[3] if( ARGV[3] and !ARGV[3].match(/^--/) )
$platform = "osx"
$platform = ARGV[2] if( ARGV[2] and !ARGV[2].match(/^--/) )
$version = "dev"
$version = ARGV[1] if( ARGV[1] and !ARGV[1].match(/^--/) )
$build_name = "sonicawe"
$build_name = ARGV[0] if( ARGV[0] and !ARGV[0].match(/^--/) )
$packagename = $build_name + "_" + $version

$zip = true
$zip = false if(ARGV.index("--nozip"))


def cuda_lib_path(name)
    return "#{$cuda_library_path}/lib#{name}.dylib"
end

def custom_lib_path(name, path = nil)
    return "#{$custom_library_path}/#{"#{path}/" if(path)}lib#{name}.dylib"
end

def compiler_lib_path(name)
    return "#{$compiler_library_path}/lib#{name}.dylib"
end

def run(cmd)
    unless system(cmd)
        puts "Error: Could not run #{cmd}"
        exit(1)
    end
end

def package_macos(app_name, version, packagename, zip = false)
    libraries = [cuda_lib_path("cufft"),
                 cuda_lib_path("cudart"),
                 cuda_lib_path("tlshook"),
                 custom_lib_path("portaudio"),
                 custom_lib_path("portaudiocpp"),
                 custom_lib_path("sndfile"),
                 custom_lib_path("FLAC"),
                 custom_lib_path("ogg"),
                 custom_lib_path("vorbis"),
                 custom_lib_path("vorbisenc"),
                 custom_lib_path("hdf5"),
                 custom_lib_path("hdf5_hl"),
                 compiler_lib_path("System.B"),
                 compiler_lib_path("stdc++.6")];

    directories = ["Contents/Frameworks",
                   "Contents/MacOS",
                   "Contents/MacOS/plugins",
                   "Contents/Resources",
                   "Contents/plugins"]

    executables = [[$custom_exec, "sonicawe"],
                   ["package-macos~/launcher", "launcher"]]

    additionals = [[$custom_exec + "-cuda", "sonicawe-cuda"]]

    resources = ["package-macos~/aweicon-project.icns",
                 "package-macos~/aweicon.icns"]

    use_bin = Array.new()

    appfolder = "sonicawe.app"
    # appfolder = "#{app_name}.app"
    run("rm -rf #{appfolder}")

    # Creating directories
    puts " Creating application directories ".center($command_line_width, "=")
    directories.each do |directory|
        puts " creating: #{appfolder}/#{directory}"
        run("mkdir -p #{appfolder}/#{directory}")
    end

    # Copying libraries
    puts " Copying dynamic libraries ".center($command_line_width, "=")
    libraries.each do |library|
        puts " copying: #{library}"
        local_lib = "#{appfolder}/Contents/Frameworks/#{File.basename(library)}"
        use_bin.push(local_lib)
        run("cp #{library} #{local_lib}")
        run("chmod +w #{local_lib}")
    end

    # Make libgcc_s a symbol reference to System.B
    run("ln -s libSystem.B.dylib #{appfolder}/Contents/Frameworks/libgcc_s.1.dylib")

    # Copying executables
    puts " Copying executables ".center($command_line_width, "=")
    executables.each do |executable|
        puts " copying: #{executable[0]}"
        local_exec = "#{appfolder}/Contents/MacOS/#{File.basename(executable[1])}"
        use_bin.push(local_exec)
        run("cp #{executable[0]} #{local_exec}")
    end

    # Copying additionals
    puts " Copying executables ".center($command_line_width, "=")
    additionals.each do |additional|
        puts " copying: #{additional[0]}"
        local_name = "#{appfolder}/Contents/MacOS/#{File.basename(additional[1])}"
        use_bin.push(local_name)
        unless system("cp #{additional[0]} #{local_name}")
            puts "Warning: Could not copy #{additional[0]}"
        end
    end

    # Copying resources
    puts " Copying resources ".center($command_line_width, "=")
    resources.each do |resource|
        puts " copying: #{resource}"
        run("cp -r #{resource} #{appfolder}/Contents/Resources/#{File.basename(resource)}")
    end
    run("cp -r ../matlab #{appfolder}/Contents/MacOS/matlab")
    run("cp -r ../plugins #{appfolder}/Contents/MacOS/plugins/examples")

    # Add application information
    puts " Adding application information ".center($command_line_width, "=")
    puts " writing: Info.plist"
    info = File.read("package-macos~/Info.plist")
    info.gsub!("(VERSION_TAG)", "#{version}")
    info.gsub!("(LONG_VERSION_TAG)", "Sonic AWE #{version}")
    File.open("#{appfolder}/Contents/Info.plist", "w") do |file|
        file.write(info)
    end
    puts " copying: package-macos~/PkgInfo"
    system("cp package-macos~/PkgInfo #{appfolder}/Contents/PkgInfo")

    # Setting install names
    puts " Fixing install names ".center($command_line_width, "=")
    libraries.each do |library|
        libname = "#{File.basename(library)}"
        puts " library: #{libname}"

        libpath = "#{appfolder}/Contents/Frameworks"
        newlibpath = "@executable_path/../Frameworks"
        libfile = "#{libpath}/#{libname}"
        targetid = `otool -DX #{libfile}`.strip
        newtargetid = "#{newlibpath}/#{libname}"

        # set id #{newtargetid} in binary #{libfile}
        system("install_name_tool -id #{newtargetid} #{libfile}")
        #run("install_name_tool -id #{newtargetid} #{libfile}")

        use_bin.each do |path|
            binary_uses_this_lib = !`otool -L #{path} | grep #{targetid}`.empty?
            next unless binary_uses_this_lib

            puts "  in binary: #{File.basename(path)}"
            # change install name for #{libpath}/#{libname} from #{targetid} to #{newtargetid} in binary #{path}
            system("install_name_tool -change #{targetid} #{newtargetid} #{path}")
            #run("install_name_tool -change #{targetid} #{newtargetid} #{path}")
        end
    end

    run("macdeployqt sonicawe.app -executable=sonicawe.app/Contents/MacOS/sonicawe-cuda")

	# macdeployqt can create a dmg with the -dmg argument, rename the resulting dmg:
    # run("mv #{app_name}.dmg #{packagename}.dmg")

    # Generating zip file
    #if( zip )
    #    puts " Packaging application ".center($command_line_width, "=")
    #    if( File.exist?("#{packagename}.zip") )
    #        puts " removing: #{packagename}.zip"
    #        system("rm #{packagename}.zip")
    #    end
    #    puts " creating: #{app_name}.zip"
    #    unless system("zip -r #{packagename}.zip #{appfolder}") && system("zip -r #{packagename}.zip  ../license")
    #        puts "Error: Unable to zip application, #{appfolder}"
    #        exit(1)
    #    end
    #end

    # create nice looking dmg
    # http://stackoverflow.com/questions/96882/how-do-i-create-a-nice-looking-dmg-for-mac-os-x-using-command-line-tools
	run("rm -rf pack")
	run("mkdir pack")
	run("cp -r \"sonicawe.app\" \"pack/Sonic AWE.app\"")
	run("ln -s /Applications pack/Applications")
	run("rm -f #{packagename}.dmg")
	run("hdiutil create -size 128m -srcfolder pack -volname \"Sonic AWE\" -fs HFS+ #{packagename}.dmg")
	run("rm -rf pack")
end

package_macos($build_name, $version, $packagename, $zip)
